#!/usr/bin/env python3
"""
Data to Knowledge Graph Conversion
==================================

Extracts data from PDFs and JSONs, converts them to a knowledge graph using
LangChain's LLMGraphTransformer, and stores in Neo4j.

This creates the static knowledge base for programmer staffing GraphRAG system.

Key fixes included:
- Resolve config output paths relative to src/ (so data/... means src/data/...)
- Ingest RFPs from JSON into Neo4j
- Ingest Projects from JSON into Neo4j
- Fix LangChain prompt templating bug by escaping curly braces in additional_instructions:
  {start_date, end_date, is_current} -> {{start_date, end_date, is_current}}

Point-3 change (important):
- Proficiency/years are modeled as PROPERTIES on Person-[:HAS_SKILL]->Skill
  instead of global Person-[:HAS_PROFICIENCY]->Proficiency nodes/edges.
  This prevents wrong/ambiguous "skill level" assignments.
"""

from dotenv import load_dotenv
load_dotenv(override=True)

import os
import asyncio
from glob import glob
from typing import List
import logging
from pathlib import Path
import toml

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
from langchain_neo4j import Neo4jGraph

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataKnowledgeGraphBuilder:
    """Builds knowledge graph from PDFs and JSONs using LangChain's LLMGraphTransformer."""

    def __init__(self, config_path: str | None = None):
        """
        Initialize the data knowledge graph builder.

        IMPORTANT:
        - config file default: src/utils/config.toml
        - relative paths in config are resolved relative to src/ (not utils/)
        """
        if config_path is None:
            config_path = str(Path(__file__).resolve().parent / "utils" / "config.toml")

        self.config_path = config_path
        self.config = self._load_config(config_path)

        self.setup_neo4j()
        self.setup_llm_transformer()

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from TOML file and resolve paths relative to src/."""
        config_file = Path(config_path).resolve()
        if not config_file.exists():
            raise ValueError(f"Configuration file not found: {config_file}")

        with config_file.open("r", encoding="utf-8") as f:
            config = toml.load(f)

        # Resolve relative output paths against src/ directory
        src_dir = Path(__file__).resolve().parent  # .../repo/src
        output = config.get("output", {})

        for k, v in list(output.items()):
            if isinstance(v, str):
                p = Path(v)
                if not p.is_absolute():
                    output[k] = str((src_dir / p).resolve())
                else:
                    output[k] = str(p.resolve())

        config["output"] = output
        return config

    def setup_neo4j(self):
        """Setup Neo4j connection."""
        try:
            self.graph = Neo4jGraph()
            logger.info("✓ Connected to Neo4j successfully")

            # Complete cleanup for fresh start
            logger.info("Performing complete Neo4j cleanup...")
            self.complete_cleanup()
            logger.info("✓ Neo4j completely cleared")

        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise

    def complete_cleanup(self):
        """Perform complete Neo4j database cleanup."""
        try:
            # Step 1: Delete all nodes and relationships
            logger.info("  - Deleting all nodes and relationships...")
            self.graph.query("MATCH (n) DETACH DELETE n")

            # Step 2: Drop all constraints
            logger.info("  - Dropping all constraints...")
            constraints = self.graph.query("SHOW CONSTRAINTS")
            for constraint in constraints:
                constraint_name = constraint.get("name", "")
                if constraint_name:
                    try:
                        self.graph.query(f"DROP CONSTRAINT {constraint_name}")
                    except Exception as e:
                        logger.debug(f"    Could not drop constraint {constraint_name}: {e}")

            # Step 3: Drop all indexes
            logger.info("  - Dropping all indexes...")
            indexes = self.graph.query("SHOW INDEXES")
            for index in indexes:
                index_name = index.get("name", "")
                if index_name and not index_name.startswith("__"):  # Skip system indexes
                    try:
                        self.graph.query(f"DROP INDEX {index_name}")
                    except Exception as e:
                        logger.debug(f"    Could not drop index {index_name}: {e}")

            # Step 4: Verify cleanup
            node_count = self.graph.query("MATCH (n) RETURN count(n) as count")[0]["count"]
            rel_count = self.graph.query("MATCH ()-[r]->() RETURN count(r) as count")[0]["count"]

            if node_count == 0 and rel_count == 0:
                logger.info("  ✓ Database completely clean")
            else:
                logger.warning(f"  ⚠ Cleanup incomplete: {node_count} nodes, {rel_count} relationships remain")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            logger.info("  - Falling back to basic cleanup...")
            self.graph.query("MATCH (n) DETACH DELETE n")

    def setup_llm_transformer(self):
        """Setup LLM and graph transformer with CV-specific schema."""
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY")
        )

        # Define CV-specific ontology
        self.allowed_nodes = [
            "Person", "Company", "University", "Skill", "Technology",
            "Project", "Certification", "Location", "JobTitle", "Industry",
            "Seniority",
            "RFP", "Requirement",
            # We keep Proficiency in allowed nodes if you want it later,
            # but we DO NOT create Person->HAS_PROFICIENCY anymore (it causes ambiguity).
            "Proficiency",
            "Region", "Timezone"
        ]

        # Define relationships with directional tuples
        # Point-3 change: remove Person->HAS_PROFICIENCY and Requirement->REQUIRES_PROFICIENCY.
        self.allowed_relationships = [
            ("Person", "WORKED_AT", "Company"),
            ("Person", "STUDIED_AT", "University"),
            ("Person", "HAS_SKILL", "Skill"),
            ("Person", "LOCATED_IN", "Location"),
            ("Person", "HOLDS_POSITION", "JobTitle"),
            ("Person", "WORKED_ON", "Project"),
            ("Person", "EARNED", "Certification"),
            ("JobTitle", "AT_COMPANY", "Company"),
            ("Project", "USED_TECHNOLOGY", "Technology"),
            ("Project", "FOR_COMPANY", "Company"),
            ("Company", "IN_INDUSTRY", "Industry"),
            ("Skill", "RELATED_TO", "Technology"),
            ("Certification", "ISSUED_BY", "Company"),
            ("University", "LOCATED_IN", "Location"),

            ("Person", "HAS_SENIORITY", "Seniority"),
            ("RFP", "REQUIRES", "Requirement"),
            ("Requirement", "REQUIRES_SKILL", "Skill"),

            ("Location", "IN_REGION", "Region"),
            ("Location", "IN_TIMEZONE", "Timezone"),
        ]

        # IMPORTANT FIX: escape curly braces used as literal examples in prompt instructions.
        # LangChain treats { ... } as template variables; to keep them literal use {{ ... }}.
        self.llm_transformer = LLMGraphTransformer(
            llm=self.llm,
            allowed_nodes=self.allowed_nodes,
            allowed_relationships=self.allowed_relationships,
            node_properties=["start_date", "end_date", "level", "years_experience"],
            relationship_properties=[
                "start_date",
                "end_date",
                "is_current",
                "level",
                "years_experience",
            ],
            strict_mode=True,
            additional_instructions="""
            Use strict seniority extraction rules:
            - If newest job title contains 'Intern' -> Seniority = Intern
            - If newest job title contains 'Junior' -> Seniority = Junior
            - If newest job title contains 'Senior' or 'Lead' -> Seniority = Senior
            - If newest job title contains 'Mid' or 'Mid-level' -> Seniority = Mid
            - Otherwise assume Mid
            
            Always create exactly ONE Seniority node per Person.
            Always create Person -> HAS_SENIORITY -> Seniority.
            
            WORK EXPERIENCE EXTRACTION RULES:
            - If a work experience entry contains a job title (e.g., "Software Engineer", "Senior Developer", etc.), create a JobTitle node.
            - If the entry contains a company name, create a Company node and connect Person -> WORKED_AT -> Company.
            - Do not confuse Company with JobTitle.
            - If the experience entry includes a date range, store it on the relationship:
                Person -[:WORKED_AT {{start_date, end_date, is_current}}]-> Company
                Person -[:HOLDS_POSITION {{start_date, end_date, is_current}}]-> JobTitle
            - If end_date is "Present" or "Now", set is_current = true and treat this job as the latest.
            - Always create both JobTitle and Company nodes if both exist in the entry.
            
            SKILLS + PROFICIENCY MODEL (IMPORTANT):
            - Every skill mentioned in the CV MUST become a Skill node and a relationship from the person:
                (Person)-[:HAS_SKILL]->(Skill)
            - If you can infer proficiency level and/or years of experience for a specific skill,
              DO NOT create separate Proficiency nodes.
              Instead, attach them as PROPERTIES on the HAS_SKILL relationship:
                (Person)-[:HAS_SKILL {{level: "Beginner|Intermediate|Advanced", years_experience: <number>}}]->(Skill)
            - Use level only when explicitly stated or strongly implied (e.g., "expert in", "advanced", "proficient").
            - years_experience MUST be a number (integer). If the text says ">5 years", set years_experience to 5. If it says "5+ years", set years_experience to 5. If unknown, omit it.
            - NEVER use strings like ">5" or "5+" for years_experience.
            
            LOCATION EXTRACTION RULES (CRITICAL):
            - Create a Location node ONLY if a location is explicitly mentioned in the CV.
            - Accept BOTH of the following formats:
                • "Location: City"
                • "Location: City, Country"
                • "Based in City"
                • "Based in City, Country"
            
            - If ONLY a city name is provided:
                → create Location.id = "<City>"
                → DO NOT guess or infer country, region, or timezone, UNLESS it is a well-known city listed below.
            
            - If both city and country are provided:
                → create ONE Location node
                → Location.id MUST be ONLY the city name
                  (e.g. "Tokyo, Japan" → Location.id = "Tokyo")
            
            - If you identify a well-known city, you MUST also create the corresponding Timezone and Region nodes and connect them:
                (Location)-[:IN_REGION]->(Region)
                (Location)-[:IN_TIMEZONE]->(Timezone)
                
                Examples:
                - "San Francisco" or "Los Angeles" or "Seattle" -> Timezone: "Pacific Time (PT)", Region: "North America"
                - "New York" or "Boston" -> Timezone: "Eastern Time (ET)", Region: "North America"
                - "London" -> Timezone: "GMT", Region: "Europe"
                - "Tokyo" -> Timezone: "JST", Region: "Asia"
            
            - If a CV mentions "Pacific Timezone" or "PT" or "PST" or "PDT" directly, ALWAYS create the Timezone node with ID "Pacific Time (PT)" and link it.
            - NEVER invent, infer, or hallucinate locations if they are not in the text.
            - NEVER split Location into multiple nodes.
            - Always create:
                (Person)-[:LOCATED_IN]->(Location)

              
            EDUCATION EXTRACTION RULES (VERY IMPORTANT):
            - If an education entry contains a university or school name, create:
                (Person)-[:STUDIED_AT]->(University)
            
            - If the education entry contains a date range, store it on the STUDIED_AT relationship:
                Person -[:STUDIED_AT {{start_date, end_date, is_current}}]-> University
            
            - Date formats may include:
                "2015 - 2019"
                "2015–2019"
                "2015 to 2019"
                "2019"
                "2019 - Present"
            
            - If only one year is provided (e.g. "2019"):
                set start_date = 2019
                omit end_date
            
            - If end date is "Present", "Now", or equivalent:
                set is_current = true
                omit end_date
            
            - If dates are not explicitly stated:
                still create STUDIED_AT
                but do NOT guess dates (leave properties empty)
            
            - Dates MUST be stored as strings (e.g. "2017", "2017-09").
            
            ==============================
            LOCATION & REGION MODEL (ADD-ONLY)
            ==============================
            
            Location semantics MUST follow this hierarchy:
            
            1. City
            2. Region (continent or macro-region)
            3. Timezone (IANA or GMT offset)
            
            NODES:
            - (:Location)        → represents a CITY ONLY (e.g. "Tokyo", "Berlin")
            - (:Region)          → represents a macro region (e.g. "Asia", "Europe")
            - (:Timezone)        → represents a timezone (e.g. "GMT+9", "Asia/Tokyo")
            
            RELATIONSHIPS:
            - (Person)-[:LOCATED_IN]->(Location)
            - (Location)-[:IN_REGION]->(Region)
            - (Location)-[:IN_TIMEZONE]->(Timezone)
            
            ABSOLUTE RULES:
            - NEVER store region names (Asia, Europe, EMEA, APAC) as Location.id
            - NEVER match "Asia", "Europe", "timezone" against Location.id
            - City names ONLY are allowed as Location.id
            
            QUERY INTERPRETATION RULES:
            
            If the user asks about:
            - "city" → filter by Location.id
            - "region" (Asia, Europe, APAC, EMEA) →
                traverse: (Person)-[:LOCATED_IN]->(Location)-[:IN_REGION]->(Region)
            - "timezone" →
                traverse: (Person)-[:LOCATED_IN]->(Location)-[:IN_TIMEZONE]->(Timezone)
            
            EXAMPLES (MANDATORY):
            
            ✔ Correct (region query):
            MATCH (p:Person)-[:LOCATED_IN]->(l:Location)-[:IN_REGION]->(r:Region)
            WHERE toLower(r.id) = "asia"
            RETURN p.id
            
            ✘ Incorrect:
            MATCH (p:Person)-[:LOCATED_IN]->(l:Location)
            WHERE l.id CONTAINS "Asia"
            
            DATA EXTRACTION RULES:
            - If CV contains only city → create Location ONLY
            - If region/timezone is NOT explicitly known → DO NOT GUESS
            - Region/Timezone nodes may be added later via enrichment scripts
            
            ANTI-HALLUCINATION:
            - "Asia", "Europe", "timezone", "Japan" are NEVER Company nodes
            - Location-related questions MUST use Location / Region / Timezone graph paths

            When the question asks for timezone (GMT+3, Japanese timezone, etc):
            - ALWAYS query using:
              (Person)-[:LOCATED_IN]->(Location)-[:IN_TIMEZONE]->(Timezone)
            - NEVER use Person-[:LOCATED_IN]->Timezone directly

            """
        )

        logger.info("✓ LLM Graph Transformer initialized with CV schema")

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text content from PDF using PyPDFLoader."""
        try:
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            full_text = "\n\n".join([page.page_content for page in pages])
            logger.debug(f"Extracted {len(full_text)} characters from {pdf_path}")
            return full_text
        except Exception as e:
            logger.error(f"Failed to extract text from {pdf_path}: {e}")
            return ""

    def normalize_seniority(self, title: str) -> str:
        title_low = (title or "").lower()
        if "intern" in title_low:
            return "Intern"
        if "junior" in title_low:
            return "Junior"
        if "lead" in title_low or "senior" in title_low:
            return "Senior"
        if "mid" in title_low:
            return "Mid"
        return "Mid"

    def get_latest_job_title(self, graph_document):
        """
        Extract latest job title from graph_document.
        Assumption: first HOLDS_POSITION relationship corresponds to newest job.
        """
        for rel in graph_document.relationships:
            if rel.type == "HOLDS_POSITION":
                return rel.target
        return None

    async def convert_cv_to_graph(self, pdf_path: str) -> List:
        """Convert a single CV PDF to graph documents."""
        logger.info(f"Processing: {Path(pdf_path).name}")

        text_content = self.extract_text_from_pdf(pdf_path)
        if not text_content.strip():
            logger.warning(f"No text extracted from {pdf_path}")
            return []

        document = Document(
            page_content=text_content,
            metadata={"source": pdf_path, "type": "cv"}
        )

        try:
            graph_documents = await self.llm_transformer.aconvert_to_graph_documents([document])
            logger.info(f"✓ Extracted graph from {Path(pdf_path).name}")

            # Normalize seniority based on latest job title
            for gd in graph_documents:
                person_nodes = [n for n in gd.nodes if n.type == "Person"]
                if not person_nodes:
                    continue
                person = person_nodes[0]

                latest_job = self.get_latest_job_title(gd)
                if not latest_job:
                    continue

                seniority_value = self.normalize_seniority(getattr(latest_job, "id", ""))

                existing_seniority_nodes = [n for n in gd.nodes if n.type == "Seniority"]
                if existing_seniority_nodes:
                    seniority_node = existing_seniority_nodes[0]
                    seniority_node.id = seniority_value
                    seniority_node.properties["id"] = seniority_value
                    seniority_node.properties["name"] = seniority_value
                else:
                    seniority_node = type(latest_job)(
                        id=seniority_value,
                        type="Seniority",
                        properties={"id": seniority_value, "name": seniority_value},
                    )
                    gd.nodes.append(seniority_node)

                # Remove any existing HAS_SENIORITY relations from this person
                gd.relationships = [
                    rel for rel in gd.relationships
                    if not (rel.type == "HAS_SENIORITY" and rel.source.id == person.id)
                ]

                # Create HAS_SENIORITY relationship
                if gd.relationships:
                    rel_cls = type(gd.relationships[0])
                    gd.relationships.append(rel_cls(
                        source=person,
                        target=seniority_node,
                        type="HAS_SENIORITY",
                        properties={}
                    ))

            return graph_documents

        except Exception as e:
            logger.error(f"Failed to convert {pdf_path} to graph: {e}")
            return []

    async def process_all_cvs(self, cv_directory: str | None = None) -> int:
        """Process all PDF CVs in the directory."""
        if cv_directory is None:
            cv_directory = self.config["output"]["programmers_dir"]

        pdf_files = glob(os.path.join(cv_directory, "*.pdf"))
        # keep cost controlled
        # pdf_files = pdf_files[:5]

        if not pdf_files:
            logger.error(f"No PDF files found in {cv_directory}")
            return 0

        logger.info(f"Found {len(pdf_files)} PDF files to process")

        processed_count = 0
        all_graph_documents = []

        for pdf_path in pdf_files:
            graph_documents = await self.convert_cv_to_graph(pdf_path)
            if graph_documents:
                all_graph_documents.extend(graph_documents)
                processed_count += 1
            else:
                logger.warning(f"Failed to process {pdf_path}")

        if all_graph_documents:
            logger.info("Storing graph documents in Neo4j...")
            self.store_graph_documents(all_graph_documents)

        return processed_count

    def store_graph_documents(self, graph_documents: List):
        """Store graph documents in Neo4j."""
        try:
            self.graph.add_graph_documents(
                graph_documents,
                baseEntityLabel=True,
                include_source=True
            )

            total_nodes = sum(len(doc.nodes) for doc in graph_documents)
            total_relationships = sum(len(doc.relationships) for doc in graph_documents)

            logger.info(f"✓ Stored {len(graph_documents)} documents in Neo4j")
            logger.info(f"✓ Total nodes: {total_nodes}")
            logger.info(f"✓ Total relationships: {total_relationships}")

            self.create_indexes()

        except Exception as e:
            logger.error(f"Failed to store graph documents: {e}")
            raise

    def create_indexes(self):
        """Create indexes for better query performance."""
        indexes = [
            "CREATE INDEX person_name IF NOT EXISTS FOR (p:Person) ON (p.id)",
            "CREATE INDEX company_name IF NOT EXISTS FOR (c:Company) ON (c.id)",
            "CREATE INDEX skill_name IF NOT EXISTS FOR (s:Skill) ON (s.id)",
            "CREATE INDEX entity_base IF NOT EXISTS FOR (e:__Entity__) ON (e.id)"
        ]

        for index_query in indexes:
            try:
                self.graph.query(index_query)
            except Exception as e:
                logger.debug(f"Index might already exist or failed: {e}")

    def validate_graph(self):
        """Validate the created knowledge graph."""
        logger.info("Validating knowledge graph...")

        queries = {
            "Total nodes": "MATCH (n) RETURN count(n) as count",
            "Total relationships": "MATCH ()-[r]->() RETURN count(r) as count",
            "Node types": "MATCH (n) RETURN labels(n)[0] as type, count(n) as count ORDER BY count DESC",
            "Relationship types": "MATCH ()-[r]->() RETURN type(r) as type, count(r) as count ORDER BY count DESC"
        }

        for description, query in queries.items():
            try:
                result = self.graph.query(query)
                if description in ["Total nodes", "Total relationships"]:
                    logger.info(f"{description}: {result[0]['count']}")
                else:
                    logger.info(f"\n{description}:")
                    for row in result[:10]:
                        logger.info(f"  {row.get('type')}: {row.get('count')}")
            except Exception as e:
                logger.error(f"Failed to execute validation query '{description}': {e}")

        sample_queries = [
            "MATCH (p:Person)-[r:HAS_SKILL]->(s:Skill) RETURN p.id AS person, s.id AS skill, r.level AS level, r.years_experience AS years LIMIT 10",
            "MATCH (p:Person)-[:WORKED_AT]->(c:Company) RETURN p.id AS person, c.id AS company LIMIT 5",
            "MATCH (p:Project)-[:REQUIRES]->(:Requirement)-[:REQUIRES_SKILL]->(s:Skill) RETURN p.id AS project, collect(s.id)[0..10] AS skills LIMIT 5",
            "MATCH (r:RFP)-[:REQUIRES]->(:Requirement)-[:REQUIRES_SKILL]->(s:Skill) RETURN r.id AS rfp, collect(s.id)[0..10] AS skills LIMIT 5",
        ]

        logger.info("\nSample relationships:")
        for query in sample_queries:
            try:
                result = self.graph.query(query)
                for row in result:
                    logger.info(f"  {dict(row)}")
            except Exception as e:
                logger.debug(f"Sample query failed: {e}")

    # -----------------------
    # RFP ingestion (JSON)
    # -----------------------
    def load_rfps_from_json(self, json_path: str):
        """Load RFPs from JSON file."""
        import json

        if not os.path.exists(json_path):
            logger.error(f"RFP JSON not found: {json_path}")
            return []

        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def convert_rfps_to_graph(self, json_path: str):
        """Convert RFP JSON into Neo4j nodes and relationships."""
        rfps = self.load_rfps_from_json(json_path)
        if not rfps:
            logger.warning("No RFPs found.")
            return

        for rfp in rfps:
            rfp_id = rfp.get("id")
            if not rfp_id:
                continue

            self.graph.query(
                """
                MERGE (r:RFP {id: $id})
                SET r.title = $title,
                    r.client = $client,
                    r.project_type = $project_type,
                    r.duration_months = $duration_months,
                    r.team_size = $team_size,
                    r.budget_range = $budget_range,
                    r.location = $location,
                    r.remote_allowed = $remote_allowed
                """,
                {
                    "id": rfp_id,
                    "title": rfp.get("title"),
                    "client": rfp.get("client"),
                    "project_type": rfp.get("project_type"),
                    "duration_months": rfp.get("duration_months"),
                    "team_size": rfp.get("team_size"),
                    "budget_range": rfp.get("budget_range"),
                    "location": rfp.get("location"),
                    "remote_allowed": rfp.get("remote_allowed"),
                },
            )

            for req in rfp.get("requirements", []):
                skill_name = req.get("skill_name")
                if not skill_name:
                    continue

                req_id = f"{rfp_id}-{skill_name}"

                self.graph.query(
                    """
                    MERGE (req:Requirement {id: $req_id})
                    SET req.min_proficiency = $min_proficiency,
                        req.is_mandatory = $is_mandatory
                    """,
                    {
                        "req_id": req_id,
                        "min_proficiency": req.get("min_proficiency"),
                        "is_mandatory": req.get("is_mandatory"),
                    },
                )

                self.graph.query(
                    "MERGE (s:Skill {id: $skill})",
                    {"skill": skill_name},
                )

                self.graph.query(
                    """
                    MATCH (r:RFP {id: $rfp_id})
                    MATCH (req:Requirement {id: $req_id})
                    MATCH (s:Skill {id: $skill})
                    MERGE (r)-[:REQUIRES]->(req)
                    MERGE (req)-[:REQUIRES_SKILL]->(s)
                    """,
                    {"rfp_id": rfp_id, "req_id": req_id, "skill": skill_name},
                )

        logger.info("✓ RFPs loaded into Neo4j successfully")

    # -----------------------
    # Projects ingestion (JSON)
    # -----------------------
    def load_projects_from_json(self, json_path: str):
        """Load Projects from JSON file."""
        import json

        if not os.path.exists(json_path):
            logger.error(f"Projects JSON not found: {json_path}")
            return []

        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def convert_projects_to_graph(self, json_path: str):
        """Convert Projects JSON into Neo4j nodes and relationships."""
        projects = self.load_projects_from_json(json_path)
        if not projects:
            logger.warning("No projects found.")
            return

        for project in projects:
            project_id = project.get("id")
            if not project_id:
                continue

            self.graph.query(
                """
                MERGE (p:Project {id: $id})
                SET p.name = $name,
                    p.client = $client,
                    p.description = $description,
                    p.start_date = $start_date,
                    p.end_date = $end_date,
                    p.estimated_duration_months = $estimated_duration_months,
                    p.budget = $budget,
                    p.status = $status,
                    p.team_size = $team_size
                """,
                {
                    "id": project_id,
                    "name": project.get("name"),
                    "client": project.get("client"),
                    "description": project.get("description"),
                    "start_date": project.get("start_date"),
                    "end_date": project.get("end_date"),
                    "estimated_duration_months": project.get("estimated_duration_months"),
                    "budget": project.get("budget"),
                    "status": project.get("status"),
                    "team_size": project.get("team_size"),
                },
            )

            client = project.get("client")
            if client:
                self.graph.query(
                    """
                    MERGE (c:Company {id: $client})
                    WITH c
                    MATCH (p:Project {id: $pid})
                    MERGE (p)-[:FOR_COMPANY]->(c)
                    """,
                    {"pid": project_id, "client": client},
                )

            for req in project.get("requirements", []):
                skill_name = req.get("skill_name")
                if not skill_name:
                    continue

                req_id = f"{project_id}-{skill_name}"

                self.graph.query(
                    """
                    MERGE (req:Requirement {id: $req_id})
                    SET req.min_proficiency = $min_proficiency,
                        req.is_mandatory = $is_mandatory
                    """,
                    {
                        "req_id": req_id,
                        "min_proficiency": req.get("min_proficiency"),
                        "is_mandatory": req.get("is_mandatory"),
                    },
                )

                self.graph.query(
                    "MERGE (s:Skill {id: $skill})",
                    {"skill": skill_name},
                )

                self.graph.query(
                    """
                    MATCH (p:Project {id: $pid})
                    MATCH (req:Requirement {id: $req_id})
                    MATCH (s:Skill {id: $skill})
                    MERGE (p)-[:REQUIRES]->(req)
                    MERGE (req)-[:REQUIRES_SKILL]->(s)
                    """,
                    {"pid": project_id, "req_id": req_id, "skill": skill_name},
                )

            # Assignments
            for assignment in project.get("assigned_programmers", []):
                programmer_name = assignment.get("programmer_name")
                if not programmer_name:
                    continue
                self.graph.query(
                    """
                    MATCH (p:Project {id: $pid})
                    MATCH (person:Person {id: $name})
                    MERGE (person)-[r:WORKED_ON]->(p)
                    SET r.assignment_start_date = $start,
                        r.assignment_end_date = $end
                    """,
                    {
                        "pid": project_id,
                        "name": programmer_name,
                        "start": assignment.get("assignment_start_date"),
                        "end": assignment.get("assignment_end_date"),
                    },
                )

        logger.info("✓ Projects loaded into Neo4j successfully")


async def main():
    """Main function to convert CVs + JSONs to knowledge graph."""
    print("Converting PDF CVs + JSONs to Knowledge Graph")
    print("=" * 50)

    try:
        builder = DataKnowledgeGraphBuilder()

        # 1) CV PDFs -> graph
        processed_count = await builder.process_all_cvs()
        if processed_count > 0:
            print(f"\n✓ Successfully processed {processed_count} CV(s)")
        else:
            print("\n⚠ No CVs were successfully processed (check PDF directory/config)")

        # 2) RFP JSON -> graph
        rfp_json_path = str(Path(builder.config["output"]["rfps_dir"]) / "rfps.json")
        builder.convert_rfps_to_graph(rfp_json_path)

        # 3) Projects JSON -> graph
        projects_json_path = str(Path(builder.config["output"]["projects_dir"]) / "projects.json")
        builder.convert_projects_to_graph(projects_json_path)

        # 4) Validate whole graph
        builder.validate_graph()

        # Next steps (best-effort)
        query_a = Path("3_query_knowledge_graph.py")
        query_b = Path("query_knowledge_graph_3.py")
        query_file = str(query_a) if query_a.exists() else (str(query_b) if query_b.exists() else "3_query_knowledge_graph.py")

        print("\n✓ Knowledge graph created in Neo4j")
        print("\nNext steps:")
        print(f"1. Run: uv run python {query_file}")
        print("2. Open Neo4j Browser to explore the graph")
        print("3. Try GraphRAG queries!")

    except Exception as e:
        logger.error(f"Failed to build knowledge graph: {e}")
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())

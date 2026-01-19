"""
Data to Knowledge Graph Conversion
==================================

Extracts data from PDFs and JSONs, converts them to a knowledge graph using
LangChain's LLMGraphTransformer, and stores in Neo4j.

This creates the static knowledge base for programmer staffing GraphRAG system.
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

from unstructured.partition.pdf import partition_pdf
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
from langchain_neo4j import Neo4jGraph

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataKnowledgeGraphBuilder:
    """Builds knowledge graph from PDFs and JSONs using LangChain's LLMGraphTransformer."""

    def __init__(self, config_path: str = "utils/config.toml"):
        """Initialize the data knowledge graph builder."""
        self.config = self._load_config(config_path)
        self.setup_neo4j()
        self.setup_llm_transformer()

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from TOML file."""
        if not os.path.exists(config_path):
            raise ValueError(f"Configuration file not found: {config_path}")

        with open(config_path, 'r') as f:
            config = toml.load(f)

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
            constraints_query = "SHOW CONSTRAINTS"
            constraints = self.graph.query(constraints_query)
            for constraint in constraints:
                constraint_name = constraint.get('name', '')
                if constraint_name:
                    try:
                        drop_query = f"DROP CONSTRAINT {constraint_name}"
                        self.graph.query(drop_query)
                        logger.debug(f"    Dropped constraint: {constraint_name}")
                    except Exception as e:
                        logger.debug(f"    Could not drop constraint {constraint_name}: {e}")

            # Step 3: Drop all indexes
            logger.info("  - Dropping all indexes...")
            indexes_query = "SHOW INDEXES"
            indexes = self.graph.query(indexes_query)
            for index in indexes:
                index_name = index.get('name', '')
                if index_name and not index_name.startswith('__'):  # Skip system indexes
                    try:
                        drop_query = f"DROP INDEX {index_name}"
                        self.graph.query(drop_query)
                        logger.debug(f"    Dropped index: {index_name}")
                    except Exception as e:
                        logger.debug(f"    Could not drop index {index_name}: {e}")

            # Step 4: Verify cleanup
            node_count = self.graph.query("MATCH (n) RETURN count(n) as count")[0]['count']
            rel_count = self.graph.query("MATCH ()-[r]->() RETURN count(r) as count")[0]['count']

            if node_count == 0 and rel_count == 0:
                logger.info("  ✓ Database completely clean")
            else:
                logger.warning(f"  ⚠ Cleanup incomplete: {node_count} nodes, {rel_count} relationships remain")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            # Fallback to basic cleanup
            logger.info("  - Falling back to basic cleanup...")
            self.graph.query("MATCH (n) DETACH DELETE n")

    def setup_llm_transformer(self):
        """Setup LLM and graph transformer with CV-specific schema."""
        # Initialize LLM - using GPT-4o-mini for cost efficiency
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
            "RFP", "Requirement", "Proficiency"
        ]

        # Define relationships with directional tuples
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
            ("Requirement", "REQUIRES_PROFICIENCY", "Proficiency"),
            ("Person", "HAS_SKILL", "Skill"),
            ("Person", "HAS_PROFICIENCY", "Proficiency")
        ]

        # Initialize transformer with strict schema
        self.llm_transformer = LLMGraphTransformer(
            llm=self.llm,
            allowed_nodes=self.allowed_nodes,
            allowed_relationships=self.allowed_relationships,
            node_properties=["start_date", "end_date", "level", "years_experience"],
            strict_mode=True,
            additional_instructions="""
            Use strict seniority extraction rules:
            - If newest job title contains 'Intern' -> Seniority = Intern
            - If newest job title contains 'Junior' -> Seniority = Junior
            - If newest job title contains 'Senior' or 'Lead' -> Seniority = Senior
            - If newest job title contains 'Mid' or 'Mid-level' -> Seniority = Mid
            - Otherwise assume Mid

            Always create exactly ONE Seniority node per Person.
            Always create Person -> HAS_SENIORITY -> Seniority
            
            
            - If a work experience entry contains a job title (e.g., "Software Engineer", "Senior Developer", etc.), create a JobTitle node.
            - If the entry contains a company name, create a Company node and connect Person -> WORKED_AT -> Company.
            - Do not confuse Company with JobTitle.
            - If the experience entry includes a date range, store it on the relationship:
                Person -[:WORKED_AT {{start_date, end_date, is_current}}]-> Company
                Person -[:HOLDS_POSITION {{start_date, end_date, is_current}}]-> JobTitle
            - If end_date is "Present" or "Now", set is_current = true and treat this job as the latest.
            - Always create both JobTitle and Company nodes if both exist in the entry.
            
            IMPORTANT — RFP MATCHING LOGIC:
            When a question asks for:
            - "best candidate"
            - "most suitable candidate"
            - "top developer"
            - "who should be assigned to this RFP"
            
            Interpret it as:
            
            The best candidate is the Person who matches the HIGHEST NUMBER of required skills
            defined by the RFP via the path:
            
            (RFP)-[:REQUIRES]->(Requirement)-[:REQUIRES_SKILL]->(Skill)
            (Person)-[:HAS_SKILL]->(Skill)
            
            Matching is CASE-INSENSITIVE on Skill.id.
            
            Ranking logic:
            - Count DISTINCT matched skills
            - Order descending by number of matched skills
            - Return the top candidate (LIMIT 1)
            
            This logic corresponds to the Cypher pattern:
            
            MATCH (r:RFP {{title: $rfp_title}})
            MATCH (r)-[:REQUIRES]->(:Requirement)-[:REQUIRES_SKILL]->(s:Skill)
            
            WITH r, collect(DISTINCT toLower(s.id)) AS requiredSkills
            
            MATCH (p:Person)-[:HAS_SKILL]->(ps:Skill)
            WITH p, requiredSkills, collect(DISTINCT toLower(ps.id)) AS personSkills
            
            WITH p, requiredSkills,
                 size(apoc.coll.intersection(requiredSkills, personSkills)) AS matchedSkills
            
            RETURN p.id AS candidate, matchedSkills
            ORDER BY matchedSkills DESC, p.id ASC
            LIMIT 1

            
            If multiple candidates are requested, return them ordered by matched skill count. If you do not find any candidates, try again.
            """
        )

        logger.info("✓ LLM Graph Transformer initialized with CV schema")

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text content from PDF using unstructured."""
        try:
            elements = partition_pdf(filename=pdf_path)
            full_text = "\n\n".join([str(element) for element in elements])
            logger.debug(f"Extracted {len(full_text)} characters from {pdf_path}")
            return full_text
        except Exception as e:
            logger.error(f"Failed to extract text from {pdf_path}: {e}")
            return ""

    def normalize_seniority(self, title: str) -> str:
        title_low = title.lower()
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
                # in your model, rel.target is the JobTitle node
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
                print(set([n.type for n in gd.nodes]))

                person_nodes = [n for n in gd.nodes if n.type == "Person"]
                if not person_nodes:
                    continue
                person = person_nodes[0]

                latest_job = self.get_latest_job_title(gd)
                if not latest_job:
                    continue

                seniority_value = self.normalize_seniority(latest_job.id)

                # Check if Seniority node already exists
                existing_seniority_nodes = [n for n in gd.nodes if n.type == "Seniority"]
                seniority_node = None

                if existing_seniority_nodes:
                    seniority_node = existing_seniority_nodes[0]
                    seniority_node.id = seniority_value
                    seniority_node.properties["id"] = seniority_value
                    seniority_node.properties["name"] = seniority_value
                else:
                    # Create Seniority node
                    seniority_node = type(latest_job)(id=seniority_value, type="Seniority",
                                                      properties={"id": seniority_value, "name": seniority_value})
                    gd.nodes.append(seniority_node)

                # Remove any existing HAS_SENIORITY relations from this person
                gd.relationships = [
                    rel for rel in gd.relationships
                    if not (rel.type == "HAS_SENIORITY" and rel.source.id == person.id)
                ]

                # Create relationship
                gd.relationships.append(type(gd.relationships[0])(
                    source=person,
                    target=seniority_node,
                    type="HAS_SENIORITY",
                    properties={}
                ))

            return graph_documents

        except Exception as e:
            logger.error(f"Failed to convert {pdf_path} to graph: {e}")
            return []


    async def process_all_cvs(self, cv_directory: str = None) -> int:
        """Process all PDF CVs in the directory."""
        if cv_directory is None:
            cv_directory = self.config['output']['programmers_dir']

        pdf_files = glob(os.path.join(cv_directory, "*.pdf"))
        pdf_files = pdf_files[:5]

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
                logger.debug(f"Created index: {index_query}")
            except Exception as e:
                logger.debug(f"Index might already exist: {e}")

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
                        if 'type' in row:
                            logger.info(f"  {row['type']}: {row['count']}")
                        else:
                            logger.info(f"  {row}")
            except Exception as e:
                logger.error(f"Failed to execute validation query '{description}': {e}")

        sample_queries = [
            "MATCH (p:Person)-[:HAS_SKILL]->(s:Skill) RETURN p.id, s.id LIMIT 5",
            "MATCH (p:Person)-[:WORKED_AT]->(c:Company) RETURN p.id, c.id LIMIT 5"
        ]

        logger.info("\nSample relationships:")
        for query in sample_queries:
            try:
                result = self.graph.query(query)
                for row in result:
                    logger.info(f"  {dict(row)}")
            except Exception as e:
                logger.debug(f"Sample query failed: {e}")

    def load_rfps_from_json(self, json_path: str):
        """Load RFPs from JSON file."""
        import json

        if not os.path.exists(json_path):
            logger.error(f"RFP JSON not found: {json_path}")
            return []

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return data

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
                req_id = f"{rfp_id}-{req.get('skill_name')}"
                skill_name = req.get("skill_name")

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
                    """
                    MERGE (s:Skill {id: $skill})
                    """,
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


async def main():
    """Main function to convert CVs to knowledge graph."""
    print("Converting PDF CVs to Knowledge Graph")
    print("=" * 50)

    try:
        builder = DataKnowledgeGraphBuilder()
        processed_count = await builder.process_all_cvs()

        if processed_count > 0:
            builder.validate_graph()
            print(f"\n✓ Successfully processed {processed_count} CV(s)")
            print("✓ Knowledge graph created in Neo4j")
            print("\nNext steps:")
            print("1. Run: uv run python 3_query_knowledge_graph.py")
            print("2. Open Neo4j Browser to explore the graph")
            print("3. Try GraphRAG queries!")
        else:
            print("❌ No CVs were successfully processed")
            print("Please check the PDF files in data/cvs_pdf/ directory")

        rfp_json_path = builder.config['output']['rfps_dir'] + "/rfps.json"
        builder.convert_rfps_to_graph(rfp_json_path)

    except Exception as e:
        logger.error(f"Failed to build knowledge graph: {e}")
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())

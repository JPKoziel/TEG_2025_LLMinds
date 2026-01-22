from src.query_knowledge_graph_3 import CVGraphRAGSystem
import json

def inspect():
    system = CVGraphRAGSystem()
    
    print("--- Labels ---")
    labels = system.graph.query("CALL db.labels()")
    print(labels)
    
    print("\n--- Relationship Types ---")
    rels = system.graph.query("CALL db.relationshipTypes()")
    print(rels)
    
    print("\n--- Node Properties ---")
    for label in [l['label'] for l in labels]:
        props = system.graph.query(f"MATCH (n:{label}) UNWIND keys(n) AS key RETURN DISTINCT key")
        print(f"{label}: {[p['key'] for p in props]}")
        
    print("\n--- Relationship Properties ---")
    for rel in [r['relationshipType'] for r in rels]:
        props = system.graph.query(f"MATCH ()-[r:{rel}]->() UNWIND keys(r) AS key RETURN DISTINCT key")
        print(f"{rel}: {[p['key'] for p in props]}")

    print("\n--- Sample Person ---")
    sample_person = system.graph.query("MATCH (p:Person) RETURN p LIMIT 1")
    print(json.dumps(sample_person, indent=2))

    print("\n--- Sample STUDIED_AT ---")
    sample_study = system.graph.query("MATCH (p:Person)-[r:STUDIED_AT]->(u:University) RETURN r LIMIT 1")
    print(json.dumps(sample_study, indent=2))

if __name__ == "__main__":
    inspect()

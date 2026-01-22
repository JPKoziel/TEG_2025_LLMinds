from src.query_knowledge_graph_3 import CVGraphRAGSystem

def check_data():
    system = CVGraphRAGSystem()
    result = system.graph.query("MATCH (p:Person)-[r:HAS_SKILL]->(s:Skill) RETURN p.id, s.id, r.years_experience LIMIT 10")
    print(result)

if __name__ == "__main__":
    check_data()

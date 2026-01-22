from src.query_knowledge_graph_3 import CVGraphRAGSystem

def debug():
    system = CVGraphRAGSystem()
    queries = [
        "List available developers in Pacific timezone",
        "Average years of experience for machine learning projects",
        "Skills distribution by graduation year",
        "Optimal team composition for FinTech RFP under budget constraints",
        "Skills gaps analysis for upcoming project pipeline"
    ]
    
    for q in queries:
        print(f"\nQ: {q}")
        result = system.qa_chain.invoke({"query": q})
        print(f"Generated Cypher: {result.get('intermediate_steps', [{}])[0].get('query')}")
        print(f"Full Context: {result.get('intermediate_steps', [{}])[1].get('context')}")
        print(f"A: {result['result']}")

if __name__ == "__main__":
    debug()

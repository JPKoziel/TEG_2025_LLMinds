from src.query_knowledge_graph_3 import CVGraphRAGSystem

def debug():
    system = CVGraphRAGSystem()
    queries = [
        "How many Python developers are available next month?",
        "Count developers with AWS certifications",
        "Find senior developers with React AND Node.js experience",
        "List available developers in Pacific timezone",
        "Average years of experience for machine learning projects",
        "Total capacity available for Q4 projects",
        "Find developers who worked together successfully",
        "Developers from same university as our top performers",
        "Who becomes available after current project ends?",
        "Skills distribution by graduation year",
        "Optimal team composition for FinTech RFP under budget constraints",
        "Skills gaps analysis for upcoming project pipeline",
        "Risk assessment: single points of failure in current assignments"
    ]
    
    for q in queries:
        print(f"\nQ: {q}")
        try:
            result = system.qa_chain.invoke({"query": q})
            cypher = result.get('intermediate_steps', [{}])[0].get('query')
            context = result.get('intermediate_steps', [{}])[1].get('context')
            print(f"Generated Cypher: {cypher}")
            print(f"Full Context: {context}")
            print(f"A: {result['result']}")
        except Exception as e:
            print(f"Error: {e}")
        print("-" * 50)

if __name__ == "__main__":
    debug()

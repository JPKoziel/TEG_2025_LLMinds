from src.query_knowledge_graph_3 import CVGraphRAGSystem

graph_rag_system = CVGraphRAGSystem()

def graph_rag_service(question: str):
    return graph_rag_system.query_graph(
        question=question
    )
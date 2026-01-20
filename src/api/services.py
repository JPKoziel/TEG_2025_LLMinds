from src.query_knowledge_graph_3 import CVGraphRAGSystem
from src.naive_rag_cv_4 import NaiveRAGSystem

graph_rag_system = CVGraphRAGSystem()

def graph_rag_service(question: str):
    return graph_rag_system.query_graph(
        question=question
    )


def naive_rag_service(question: str):
    naive_rag = NaiveRAGSystem(config_path="src/utils/config.toml")
    naive_rag.initialize_system()

    return naive_rag.query(
        question=question
    )

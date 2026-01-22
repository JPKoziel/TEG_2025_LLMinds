from neo4j import GraphDatabase
import sys

def test_connection():
    uri = "bolt://localhost:7687"
    user = "neo4j"
    password = "password123"
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session() as session:
            result = session.run("RETURN 1 as n")
            record = result.single()
            print(f"Connection successful: {record['n']}")
        driver.close()
    except Exception as e:
        print(f"Connection failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_connection()

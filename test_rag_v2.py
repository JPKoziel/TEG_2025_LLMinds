import requests
import json

def test_queries():
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
    
    url = "http://localhost:8000/rag/graph"
    
    results = []
    for q in queries:
        print(f"Testing: {q}")
        try:
            response = requests.post(url, json={"question": q}, timeout=60)
            if response.status_code == 200:
                data = response.json()
                print(f"A: {data['answer']}")
                results.append({"question": q, "answer": data['answer'], "success": True})
            else:
                print(f"Error: {response.status_code} - {response.text}")
                results.append({"question": q, "error": response.text, "success": False})
        except Exception as e:
            print(f"Exception: {e}")
            results.append({"question": q, "error": str(e), "success": False})
        print("-" * 20)
        
    with open("test_results_v2.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    test_queries()

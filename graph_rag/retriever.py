from ollama_llm import generate_completion
from graph_builder import KnowledgeGraph
import json

def extract_entities_from_question(question):
    """Extract entities reliably using Phi3."""
    
    prompt = f"""
You are an information extraction system.

Extract ALL important entities from the question.

Entities may include:
- companies
- AI models
- technologies
- concepts
- organizations

Rules:
- Always include concepts (AI, Machine Learning, etc.)
- Return ONLY JSON list
- No explanation

Examples:
Question: What is AI?
Output: ["Artificial Intelligence"]

Question: What does Google DeepMind focus on?
Output: ["Google DeepMind"]

Question: Explain machine learning
Output: ["Machine Learning"]

Question: {question}
"""

    response = generate_completion(prompt)

    cleaned = response.strip()
    print(f"[Debug] Raw Phi-3 Output: {cleaned}")
    
    import re
    # Try finding an array
    match = re.search(r'\[.*\]', cleaned, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, list):
                return data
        except Exception as e:
            print(f"[Debug] JSON parse failed: {e}")
            pass
            
    # Try finding an object if Ollama force-returned a dict
    match_dict = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if match_dict:
        try:
            data = json.loads(match_dict.group(0))
            for k, v in data.items():
                if isinstance(v, list):
                    return v
        except:
            pass

    # Ultimate fallback: just find quoted strings
    fallback_entities = re.findall(r'"([^"]+)"', cleaned)
    if fallback_entities:
        return fallback_entities
        
    return []

def retrieve_context(question):
    kg = KnowledgeGraph()
    entities = extract_entities_from_question(question)

    print(f"[Retriever] Identified entities: {entities}")

    if not entities:
        kg.close()
        return "No relevant entities extracted."

    paths = []

    for entity in entities:
        cypher = """
        MATCH (n)-[r]-(m)
WHERE toLower(n.id) CONTAINS toLower($entity)
RETURN n.id AS source, type(r) AS relationship, m.id AS target
LIMIT 20

        """

        results = kg.query_graph(cypher, {"entity": entity})

        for record in results:
            paths.append(
                f"{record['source']} -[{record['relationship']}]-> {record['target']}"
            )

    kg.close()

    unique_paths = list(set(paths))

    if not unique_paths:
        return "No connected nodes found."

    return "Graph Context:\n" + "\n".join(unique_paths)
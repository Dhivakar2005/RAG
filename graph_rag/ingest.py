import json
from ollama_llm import generate_completion
from graph_builder import KnowledgeGraph

def chunk_text(text, chunk_size=300):
    """Splits a large text into manageable chunk sizes for the LLM."""
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def clean_json_response(response_text):
    """Cleans up the text output if the LLM surrounds JSON with markdown block formatting."""
    response_text = response_text.strip()
    if response_text.startswith("```json"):
        response_text = response_text[7:]
    if response_text.startswith("```"):
        response_text = response_text[3:]
    if response_text.endswith("```"):
        response_text = response_text[:-3]
    return response_text.strip()

def extract_entities_and_relationships(chunk):
    """Asks the LLM to extract Entities and Relationships and structure them as JSON."""
    prompt = f'''
    You are an AI that extracts information from text as a knowledge graph.
    Extract the main entities and the relationships between them from the following text.
    Entities should have an "id" (name) and a "label" (e.g., Person, Concept, Company, Technology).
    Relationships should have a "source" (entity id), "target" (entity id), and a "type" (e.g., DEVELOPS, USES, IS_A).
    
    Return ONLY a valid JSON object with the following structure:
    {{
      "entities": [
        {{"id": "Entity Name", "label": "Label"}}
      ],
      "relationships": [
        {{"source": "Source Entity", "target": "Target Entity", "type": "RELATION_TYPE"}}
      ]
    }}
    
    Text: {chunk}
    '''
    
    response = generate_completion(prompt, format_json=True)
    raw_json = clean_json_response(response)
    
    try:
        return json.loads(raw_json)
    except json.JSONDecodeError as e:
        print(f"Failed to parse LLM output as JSON: {e}")
        print("Raw response:", raw_json)
        return {"entities": [], "relationships": []}

def run_ingestion(filepath):
    """Orchestrates reading the text, extracting graphs, and inserting into Neo4j."""
    print(f"Reading file: {filepath}")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        print("Knowledge file not found. Please create 'data/knowledge.txt'")
        return

    kg = KnowledgeGraph()
    chunks = chunk_text(text)
    
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}...")
        data = extract_entities_and_relationships(chunk)
        
        entities = data.get("entities", [])
        relationships = data.get("relationships", [])
        
        print(f"Found {len(entities)} entities and {len(relationships)} relationships.")
        
        for ent in entities:
            if "id" in ent and "label" in ent:
                # Handle cases where the LLM might return a list instead of a string
                raw_id = ent["id"]
                raw_label = ent["label"]
                
                ent_id = str(raw_id[0] if isinstance(raw_id, list) and len(raw_id) > 0 else raw_id).strip()
                ent_label = str(raw_label[0] if isinstance(raw_label, list) and len(raw_label) > 0 else raw_label).strip().replace(" ", "_")
                
                if ent_id and ent_label:
                    kg.create_entity(ent_id, ent_label)
                
        for rel in relationships:
            if "source" in rel and "target" in rel and "type" in rel:
                raw_source = rel["source"]
                raw_target = rel["target"]
                raw_type = rel["type"]
                
                source = str(raw_source[0] if isinstance(raw_source, list) and len(raw_source) > 0 else raw_source).strip()
                target = str(raw_target[0] if isinstance(raw_target, list) and len(raw_target) > 0 else raw_target).strip()
                rel_type = str(raw_type[0] if isinstance(raw_type, list) and len(raw_type) > 0 else raw_type).strip().upper().replace(" ", "_")
                
                if source and target and rel_type:
                    kg.create_relationship(source, target, rel_type)

    print("Ingestion completed successfully!")
    kg.close()

if __name__ == "__main__":
    run_ingestion("data/knowledge.txt")

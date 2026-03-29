from retriever import retrieve_context
from ollama_llm import generate_completion

def answer_question(question):
    """Combines retrieved graph logic and the LLM generation into a final answer."""
    print("\n[Query] Retrieving context from Neo4j graph...")
    context = retrieve_context(question)
    print(f"[Query] Context found:\n{context}\n")
    
    prompt = f"""
    You are a helpful GraphRAG assistant.

    You should prioritize answering using the provided GRAPH CONTEXT.
    However, if the GRAPH CONTEXT does not contain enough information to answer the question, you may use your own general knowledge to answer it.

    GRAPH CONTEXT:
    {context}

    USER QUESTION:
    {question}

    ANSWER:
    """
    
    print("[Query] Generating answer with Phi-3 Mini...")
    answer = generate_completion(prompt)
    return answer

def chat_loop():
    """Starts the CLI interactive loop allowing user to query the GraphRAG System."""
    print("==================================================")
    print(" Local GraphRAG Assistant (Neo4j + Phi-3 + Ollama)")
    print("==================================================")
    print("Type 'exit' to quit.\n")
    
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break
            
            if not user_input.strip():
                continue
                
            response = answer_question(user_input)
            print(f"\nAssistant:\n{response}\n")
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    chat_loop()

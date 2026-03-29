from graph_builder import KnowledgeGraph

kg = KnowledgeGraph()
print("--- ALL NODES ---")
nodes = kg.query_graph("MATCH (n) RETURN n.id as id, labels(n) as labels")
for n in nodes:
    print(f"Node: {n['id']} (Label: {n['labels'][0] if n['labels'] else 'None'})")

print("\n--- ALL RELATIONSHIPS ---")
rels = kg.query_graph("MATCH (n)-[r]->(m) RETURN n.id as source, type(r) as type, m.id as target")
for r in rels:
    print(f"{r['source']} -[{r['type']}]-> {r['target']}")

kg.close()

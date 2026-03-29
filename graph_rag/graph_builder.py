from neo4j import GraphDatabase
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

class KnowledgeGraph:
    """Class to handle connections and query executions on Neo4j."""
    
    def __init__(self):
        try:
            self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
            self.driver.verify_connectivity()
            print("Successfully connected to Neo4j!")
        except Exception as e:
            print(f"Failed to connect to Neo4j: {e}")
            self.driver = None

    def close(self):
        if self.driver:
            self.driver.close()

    def create_entity(self, entity_id, entity_label):
        """Creates a node (entity) in the Neo4j graph using a parameter-safe query."""
        if not self.driver: return
        
        # Note: Cypher does not allow parameterization of labels, so we are inserting the label securely
        # Make sure entity_label doesn't contain spaces and uses alphabets
        clean_label = "".join(c for c in entity_label if c.isalnum() or c == "_")
        query = f"MERGE (n:`{clean_label}` {{id: $id}})"
        
        with self.driver.session() as session:
            session.run(query, id=entity_id)

    def create_relationship(self, source_id, target_id, relation_type):
        """Creates a relationship between two existing nodes."""
        if not self.driver: return
        
        clean_type = "".join(c for c in relation_type if c.isalnum() or c == "_")
        query = f'''
        MATCH (a {{id: $source_id}})
        MATCH (b {{id: $target_id}})
        MERGE (a)-[r:`{clean_type}`]->(b)
        '''
        
        with self.driver.session() as session:
            session.run(query, source_id=source_id, target_id=target_id)
            
    def query_graph(self, cypher_query, parameters=None):
        """Helper to run a Custom Cypher query to retrieve information from Neo4j."""
        if not self.driver: return []
        
        with self.driver.session() as session:
            result = session.run(cypher_query, parameters or {})
            return [record.data() for record in result]

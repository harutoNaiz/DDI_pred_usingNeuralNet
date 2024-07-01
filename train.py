import numpy as np
import torch
from rdkit import Chem
import torch.nn as nn
import torch.optim as optim
import joblib
import json
from torch.utils.data import DataLoader, TensorDataset
from neo4j import GraphDatabase
from node2vec import Node2Vec
import networkx as nx

# [Your existing DrugInteractionModel class and other functions here]

def create_graph_from_neo4j(uri, user, password):
    driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def get_graph(tx):
        query = """
        MATCH (d1:Drug)-[r:INTERACTS_WITH]->(d2:Drug)
        RETURN id(d1) AS source, id(d2) AS target, r.interaction AS interaction
        """
        result = tx.run(query)
        return [(record["source"], record["target"], record["interaction"]) for record in result]
    
    with driver.session() as session:
        edges = session.read_transaction(get_graph)
    
    G = nx.Graph()
    G.add_weighted_edges_from(edges)
    
    return G

def generate_node2vec_embeddings(G, dimensions=64, walk_length=30, num_walks=200, workers=4):
    node2vec = Node2Vec(G, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, workers=workers)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    
    # Generate embeddings for all nodes
    node_embeddings = {node: model.wv[node] for node in G.nodes()}
    
    return node_embeddings

def train_model_with_embeddings(preprocessed_data_path, model_path, config_path, neo4j_uri, neo4j_user, neo4j_password, batch_size=32):
    # Load preprocessed data
    print("Loading preprocessed data...")
    data = np.load(preprocessed_data_path)
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    
    print("Generating Node2Vec embeddings...")
    G = create_graph_from_neo4j(neo4j_uri, neo4j_user, neo4j_password)
    node_embeddings = generate_node2vec_embeddings(G)
    
    # Combine original features with Node2Vec embeddings
    def combine_features(X, embeddings):
        combined = []
        for x in X:
            drug_id = int(x[0])  # Assuming the first feature is the drug ID
            if drug_id in embeddings:
                combined.append(np.concatenate([x, embeddings[drug_id]]))
            else:
                combined.append(np.concatenate([x, np.zeros(64)]))  # Use zero vector if embedding not found
        return np.array(combined)
    
    X_train = combine_features(X_train, node_embeddings)
    X_test = combine_features(X_test, node_embeddings)
    
    # Update input size in config
    with open(config_path, 'r') as f:
        config = json.load(f)
    config['input_size'] += 64  # Add 64 for Node2Vec embedding dimension
    with open(config_path, 'w') as f:
        json.dump(config, f)
    
    # Continue with your existing training code
    # [Rest of your train_model function here]

if __name__ == "__main__":
    preprocessed_data_path = 'preprocessed_data.npz'
    model_path = 'drug_interaction_model.pth'
    config_path = 'model_config.json'
    neo4j_uri = "bolt://localhost:7687"
    neo4j_user = "neo4j"
    neo4j_password = "password"
    train_model_with_embeddings(preprocessed_data_path, model_path, config_path, neo4j_uri, neo4j_user, neo4j_password)
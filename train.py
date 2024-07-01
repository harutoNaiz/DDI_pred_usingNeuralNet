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

# Define the model class
class DrugInteractionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DrugInteractionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def extract_and_pad_features(smiles, max_length):
    mol = Chem.MolFromSmiles(smiles)
    
    features = []
    
    if mol is not None:
        for atom in mol.GetAtoms():
            feature_vector = [
                atom.GetAtomicNum(),
                atom.GetDegree(),
                atom.GetImplicitValence(),
                atom.GetFormalCharge(),
                atom.GetNumRadicalElectrons(),
                int(atom.GetHybridization()),
                int(atom.GetIsAromatic())
            ]
            features.extend(feature_vector)
        
        # Pad features to max_length
        features = features + [0] * ((max_length * 7) - len(features))  # Assuming 7 is the number of features per atom
    else:
        features = [0] * (max_length * 7)  # Padding for max_length * number of features per atom
    
    return features

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

def train_model(preprocessed_data_path, model_path, config_path, neo4j_uri=None, neo4j_user=None, neo4j_password=None, batch_size=32):
    # Load preprocessed data
    print("Loading preprocessed data...")
    data = np.load(preprocessed_data_path)
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    
    print("Preprocessed data loaded.")
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    input_size = config['input_size']
    hidden_size = config['hidden_size']
    output_size = config['output_size']
    
    # If Neo4j credentials are provided, generate and incorporate Node2Vec embeddings
    if neo4j_uri and neo4j_user and neo4j_password:
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
        
        # Update input size
        input_size += 64  # Add 64 for Node2Vec embedding dimension
    
    print("Initializing the model...")
    # Initialize the model
    model = DrugInteractionModel(input_size, hidden_size, output_size)
    
    # Check if CUDA is available and set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Print the device name
    if device.type == 'cuda':
        print(f'Using device: {device} ({torch.cuda.get_device_name(device)})')
    else:
        print(f'Using device: {device}')
    
    # Use DataParallel to use multiple GPUs if available
    if torch.cuda.device_count() > 1:
        print(f'Using {torch.cuda.device_count()} GPUs for training.')
        model = nn.DataParallel(model)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    
    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    print("Starting model training...")
    # Training the model
    num_epochs = 300
    for epoch in range(num_epochs):
        model.train()
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
            torch.cuda.empty_cache()  # Clear cache to free memory
    
    print("Model training complete.")
    
    # Save the trained model
    torch.save(model.state_dict(), model_path)
    print(f"Trained model saved as '{model_path}'.")

    # Evaluate the model
    print("Evaluating the model...")
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

    accuracy = correct / total * 100
    print(f'Test Accuracy: {accuracy:.2f}%')

if __name__ == "__main__":
    preprocessed_data_path = 'preprocessed_data.npz'
    model_path = 'drug_interaction_model.pth'
    config_path = 'model_config.json'
    
    # Uncomment and fill in the following lines to use Node2Vec embeddings
    # neo4j_uri = "bolt://localhost:7687"
    # neo4j_user = "neo4j"
    # neo4j_password = "password"
    # train_model(preprocessed_data_path, model_path, config_path, neo4j_uri, neo4j_user, neo4j_password)
    
    # Or use the original version without Node2Vec
    train_model(preprocessed_data_path, model_path, config_path)
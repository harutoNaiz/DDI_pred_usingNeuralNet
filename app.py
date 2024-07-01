from flask import Flask, request, render_template, jsonify, session, redirect, url_for
from py2neo import Graph
from py2neo import Node, Relationship
import bcrypt
import joblib
import json

from rdkit import Chem
from train import DrugInteractionModel, extract_and_pad_features
import pandas as pd
import torch
import numpy as np

app = Flask(__name__)
app.secret_key = 'tushar' #secreat key

# Neo4j database connection
graph = Graph("bolt://localhost:7687", auth=("neo4j", "Tushar@123"))

# Load the CSV file containing drug SMILES
df = pd.read_csv("data/drug_smiles.csv")

# Function to hash passwords
def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

# Function to predict drug interactions
def predict_interaction(model_path, encoder_path, config_path, csv_path, smiles1, smiles2):
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    input_size = config['input_size']
    hidden_size = config['hidden_size']
    output_size = config['output_size']
    max_smiles_length = config['max_smiles_length']
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DrugInteractionModel(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    label_encoder = joblib.load(encoder_path)
    df = pd.read_csv(csv_path)
    
    features1 = extract_and_pad_features(smiles1, max_smiles_length)
    features2 = extract_and_pad_features(smiles2, max_smiles_length)
    combined_features = features1 + features2
    X = np.array(combined_features, dtype=np.float32).reshape(1, -1)
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        outputs = model(X_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_prob, predicted_class = torch.max(probabilities, dim=1)
    
    predicted_interaction_type = label_encoder.inverse_transform([predicted_class.item()])[0]
    predicted_probability = float(predicted_prob.item())
    interaction_info = df[df['Interaction_type'] == predicted_interaction_type]['Description'].values.tolist()
    
    return predicted_probability, predicted_interaction_type, interaction_info

# Function to retrieve SMILES representation of a drug based on drug_id
def get_smiles(drug_id):
    row = df[df['drug_id'] == drug_id]
    if not row.empty:
        return row['smiles'].values[0]
    else:
        return None  # Handle case where drug_id is not found

# @app.route('/')
# def home_page():
#     return render_template('home.html')
@app.route('/')
def home_page():
    return redirect(url_for('signup'))

@app.route('/signin', methods=['GET', 'POST'])
def signin():
    if request.method == 'GET':
        return render_template('signin.html')
    
    data = request.json
    username = data['username']
    password = data['password']

    user = graph.run("MATCH (u:User {username: $username}) RETURN u", username=username).data()
    if user:
        user = user[0]['u']
        if bcrypt.checkpw(password.encode('utf-8'), user['password'].encode('utf-8')):
            session['username'] = username
            return jsonify({"success": True, "message": "Sign in successful"})
        else:
            return jsonify({"success": False, "message": "Invalid password"})
    else:
        return jsonify({"success": False, "message": "User not found"})

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'GET':
        return render_template('signup.html')
    
    data = request.json
    username = data['username']
    password = data['password']

    hashed_password = hash_password(password)

    # Check if user already exists
    existing_user = graph.run("MATCH (u:User {username: $username}) RETURN u", username=username).data()
    if existing_user:
        return jsonify({"success": False, "message": "User already exists!"})

    # Create new user node in Neo4j
    graph.run("CREATE (u:User {username: $username, password: $password})", username=username, password=hashed_password)
    return jsonify({"success": True, "message": "User created successfully!"})

@app.route('/predict', methods=['POST'])
def predict():
    if 'username' not in session:
        return redirect(url_for('home_page'))

    data = request.json

    drug1 = data['drug1']
    drug2 = data['drug2']

    smiles1 = get_smiles(drug1)
    smiles2 = get_smiles(drug2)

    model_path = 'states/drug_interaction_model.pth'
    encoder_path = 'states/label_encoder.joblib'
    config_path = 'states/model_config.json'
    csv_path = 'data/Interaction_information.csv'
    
    probability, interaction_type, info = predict_interaction(model_path, encoder_path, config_path, csv_path, smiles1, smiles2)
    
    # Ensure that all types are serializable
    probability = float(probability)
    interaction_type = str(interaction_type)
    info = [str(i) for i in info]

    username = session['username']
    
    # Check if relationships already exist
    existing_relationships = graph.run("""
        MATCH (u:User {username: $username})-[r:TAKES]->(d:Drug)
        WHERE d.drug_id IN [$drug1, $drug2]
        RETURN r
    """, username=username, drug1=drug1, drug2=drug2).data()

    if not existing_relationships:
        # Store drugs and their interaction in the database
        graph.run("""
            MERGE (u:User {username: $username})
            MERGE (d1:Drug {drug_id: $drug1})
            MERGE (d2:Drug {drug_id: $drug2})
            MERGE (u)-[:TAKES]->(d1)
            MERGE (u)-[:TAKES]->(d2)
            MERGE (d1)-[r:INTERACTS_WITH {interaction_type: $interaction_type, probability: $probability, info: $info}]->(d2)
        """, username=username, drug1=drug1, drug2=drug2, interaction_type=interaction_type, probability=probability, info=info)
    
    return jsonify({
        'probability': probability,
        'interaction_type': interaction_type,
        'info': info
    })

@app.route('/predict')
def predict_page():
    if 'username' not in session:
        return redirect(url_for('home_page'))
    return render_template('predict.html')

@app.route('/admin', methods=['GET'])
def admin():
    return render_template('admin.html')

@app.route('/admin-auth', methods=['POST'])
def admin_auth():
    data = request.json
    password = data.get('password')  # Use get to safely access JSON data

    # Replace 'admin_password' with your actual admin password
    if password == 'tushar':
        return jsonify({"success": True})
    else:
        return jsonify({"success": False})

        
@app.route('/get_graph_data', methods=['GET'])
def get_graph_data():
    query = """
    MATCH (u:User)-[r:TAKES]->(d:Drug)
    OPTIONAL MATCH (d)-[i:INTERACTS_WITH]->(d2:Drug)
    RETURN u, r, d, i, d2
    """
    results = graph.run(query).data()

    nodes = {}
    links = []

    for record in results:
        user = record['u']
        drug = record['d']
        interaction = record.get('i')
        drug2 = record.get('d2')

        if user['username'] not in nodes:
            nodes[user['username']] = {'id': user['username'], 'label': user['username'], 'type': 'User'}

        if drug['drug_id'] not in nodes:
            nodes[drug['drug_id']] = {'id': drug['drug_id'], 'label': drug['drug_id'], 'type': 'Drug'}

        links.append({'source': user['username'], 'target': drug['drug_id']})

        if interaction and drug2:
            if drug2['drug_id'] not in nodes:
                nodes[drug2['drug_id']] = {'id': drug2['drug_id'], 'label': drug2['drug_id'], 'type': 'Drug'}
            links.append({'source': drug['drug_id'], 'target': drug2['drug_id'], 'label': interaction['interaction_type']})

    return jsonify({'nodes': list(nodes.values()), 'links': links})

    
if __name__ == "__main__":
    app.run(port=5001, debug=True)

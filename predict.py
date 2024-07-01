import torch
import numpy as np
import argparse
from rdkit import Chem
import joblib
import json
import torch.nn as nn
import pandas as pd
from train import DrugInteractionModel,extract_and_pad_features

def predict_interaction(model_path, encoder_path, config_path, csv_path, smiles1, smiles2):
    # Load the configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    input_size = config['input_size']
    hidden_size = config['hidden_size']
    output_size = config['output_size']
    max_smiles_length = config['max_smiles_length']
    
    # Load the saved model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DrugInteractionModel(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Load the label encoder
    label_encoder = joblib.load(encoder_path)
    
    # Load information from CSV
    df = pd.read_csv(csv_path)
    
    # Extract and pad features
    features1 = extract_and_pad_features(smiles1, max_smiles_length)
    features2 = extract_and_pad_features(smiles2, max_smiles_length)
    
    # Combine features for both SMILES into a single feature vector
    combined_features = features1 + features2
    X = np.array(combined_features, dtype=np.float32).reshape(1, -1)
    
    # Convert to PyTorch tensor
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(X_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_prob, predicted_class = torch.max(probabilities, dim=1)
    
    # Decode the predicted class
    predicted_interaction_type = label_encoder.inverse_transform([predicted_class.item()])[0]
    predicted_probability = predicted_prob.item()
    df = pd.read_csv(csv_path)
    # Get information from CSV based on predicted interaction type
    interaction_info = df[df['Interaction_type'] == predicted_interaction_type]['Description'].values
    
    return predicted_probability, predicted_interaction_type, interaction_info

if __name__ == "__main__":
    path = 'data/Interaction_information.csv'
    parser = argparse.ArgumentParser(description="Predict drug-drug interaction.")
    parser.add_argument('--smiles1', type=str, required=True, help="SMILES string for the first drug.")
    parser.add_argument('--smiles2', type=str, required=True, help="SMILES string for the second drug.")
    parser.add_argument('--model_path', type=str, default='states/drug_interaction_model.pth', help="Path to the trained model.")
    parser.add_argument('--encoder_path', type=str, default='states/label_encoder.joblib', help="Path to the label encoder.")
    parser.add_argument('--config_path', type=str, default='states/model_config.json', help="Path to the model configuration.")
    parser.add_argument('--csv_path', type=str, default='data/Interaction_information.csv', help="Path to the CSV file containing interaction information.")
    
    args = parser.parse_args()
    
    probability, interaction_type, info = predict_interaction(args.model_path, args.encoder_path, args.config_path, args.csv_path, args.smiles1, args.smiles2)
    
    print(f"Predicted Probability of Interaction: {probability:.4f}")
    print(f"Predicted Interaction Type: {interaction_type}")
    if len(info) > 0:
        print(f"Information for Interaction Type {interaction_type}:")
        for i, item in enumerate(info, start=1):
            print(f"  {i}. {item}")

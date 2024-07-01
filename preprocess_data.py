import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from rdkit import Chem
import joblib
import json

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

def preprocess_data(data_path, output_data_path, encoder_path, config_path):
    # Load data from CSV
    print("Loading data from CSV...")
    data = pd.read_csv(data_path)
    
    # Extract relevant columns
    smiles1_list = data['smiles1'].tolist()
    smiles2_list = data['smiles2'].tolist()
    interaction_types = data['type'].astype(int).tolist()
    
    print("Data loaded. Number of entries:", len(data))
    
    # Verify data consistency
    invalid_indices = []
    for i, (smiles1, smiles2) in enumerate(zip(smiles1_list, smiles2_list)):
        if not isinstance(smiles1, str) or not isinstance(smiles2, str):
            invalid_indices.append(i)
    
    if invalid_indices:
        print(f"Invalid SMILES strings found at indices: {invalid_indices}")
        # Handle or remove invalid entries
        exit(1)
    
    # Calculate maximum SMILES length
    max_smiles_length = max(max(len(smiles1), len(smiles2)) for smiles1, smiles2 in zip(smiles1_list, smiles2_list))
    print("Maximum SMILES length:", max_smiles_length)
    
    # Prepare data for training
    X = []
    y = []
    
    for i, (smiles1, smiles2, interaction_type) in enumerate(zip(smiles1_list, smiles2_list, interaction_types)):
        print(f"Processing entry {i+1}/{len(data)}...")
        # Extract and pad features for each SMILES string separately
        features1 = extract_and_pad_features(smiles1, max_smiles_length)
        features2 = extract_and_pad_features(smiles2, max_smiles_length)
        
        # Combine features for both SMILES into a single feature vector
        combined_features = features1 + features2
        X.append(combined_features)
        y.append(interaction_type)
    
    print("Feature extraction and padding complete.")
    
    # Convert to numpy arrays with float16
    X = np.array(X, dtype=np.float16)
    
    # Use LabelEncoder to encode interaction_types to integer labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(interaction_types)
    output_size = len(label_encoder.classes_)  # Number of unique classes
    
    # Save the LabelEncoder
    joblib.dump(label_encoder, encoder_path)
    
    print("Data conversion to numpy arrays complete.")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Data split into training and testing sets.")
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    
    # Save preprocessed data
    np.savez(output_data_path, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    print(f"Preprocessed data saved to '{output_data_path}'.")
    
    # Save the configuration
    config = {
        'input_size': X.shape[1],
        'hidden_size': 128,
        'output_size': output_size,
        'max_smiles_length': max_smiles_length
    }
    with open(config_path, 'w') as f:
        json.dump(config, f)
    print(f"Model configuration saved as '{config_path}'.")

if __name__ == "__main__":
    data_path = 'data/ddi_training.csv'
    output_data_path = 'preprocessed_data.npz'
    encoder_path = 'label_encoder.joblib'
    config_path = 'model_config.json'
    preprocess_data(data_path, output_data_path, encoder_path, config_path)

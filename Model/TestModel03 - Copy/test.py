# Generated from: main.ipynb
# Converted at: 2026-01-26T13:13:49.935Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from pprint import pprint
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tabulate import tabulate
import asyncio
import nest_asyncio

nest_asyncio.apply()

import os
import random

config = {
    "dataset":{
        "dti":"../../Data/scope_onside_common_v3.parquet",
        "adr":"../../Data/final_rxnorm_meddra_v2.parquet"
    },
    "protein_emb_1":{
        "path":  "../../Data/3. Protein_enbeddings/ESM_embeddings_(t33_650m model).parquet",
        "id_col": "id", 
        "emb_col": "embedding"
    },
    "protein_emb_2":{
        "path": "../../Data/3. Protein_enbeddings/GVP-GNN_protein_embeddings.parquet",
        "id_col": "uniprot_id", 
        "emb_col": "embedding"
    },
    "drug_emb_1":{
        "path": "../../Data/2. Drug_embeddings/EGNN_drug_embeddings_v2.parquet", 
        "id_col": "drug_chembl_id", 
        "emb_col": "embedding"
    },
    "drug_emb_2":{
        "path": "../../Data/2. Drug_embeddings/smiles_embeddings_chemberta.parquet", 
        "id_col": "drug_chembl_id", 
        "emb_col": "embedding"
    }
}

dti_df = pd.read_parquet(config["dataset"]["dti"])
print(dti_df.info())

# copy selected columns to a new df
# drug_chembl_id as drug_id and target_uniprot_id as protein_id
dti_df = dti_df.rename(columns={"drug_chembl_id": "drug_id", "target_uniprot_id": "protein_id"})

df = dti_df.copy()
df = df[["drug_id", "protein_id", "label", "rxcui"]]


if config["protein_emb_1"]["path"]:
    protein_emb_1_df = pd.read_parquet(config["protein_emb_1"]["path"])
    protein_emb_1_df = protein_emb_1_df.rename(columns={config["protein_emb_1"]["id_col"]: "protein_id"})
    df = df.merge(protein_emb_1_df[["protein_id", config["protein_emb_1"]["emb_col"]]], on="protein_id", how="left")
    df = df.rename(columns={config["protein_emb_1"]["emb_col"]: "prot_emb_1"})

if config["protein_emb_2"]["path"]:
    protein_emb_2_df = pd.read_parquet(config["protein_emb_2"]["path"])
    protein_emb_2_df = protein_emb_2_df.rename(columns={config["protein_emb_2"]["id_col"]: "protein_id"})
    df = df.merge(protein_emb_2_df[["protein_id", config["protein_emb_2"]["emb_col"]]], on="protein_id", how="left")
    df = df.rename(columns={config["protein_emb_2"]["emb_col"]: "prot_emb_2"})

if config["drug_emb_1"]["path"]:
    drug_emb_1_df = pd.read_parquet(config["drug_emb_1"]["path"])
    drug_emb_1_df = drug_emb_1_df.rename(columns={config["drug_emb_1"]["id_col"]: "drug_id"})
    df = df.merge(drug_emb_1_df[["drug_id", config["drug_emb_1"]["emb_col"]]], on="drug_id", how="left")
    df = df.rename(columns={config["drug_emb_1"]["emb_col"]: "drug_emb_1"})

if config["drug_emb_2"]["path"]:
    drug_emb_2_df = pd.read_parquet(config["drug_emb_2"]["path"])
    drug_emb_2_df = drug_emb_2_df.rename(columns={config["drug_emb_2"]["id_col"]: "drug_id"})
    df = df.merge(drug_emb_2_df[["drug_id", config["drug_emb_2"]["emb_col"]]], on="drug_id", how="left")
    df = df.rename(columns={config["drug_emb_2"]["emb_col"]: "drug_emb_2"})



print(df.info())




class ADRData:
    def __init__(self, id_to_name_dict):
        """
        Initializes with a dictionary of {meddra_id: meddra_name}.
        """
        self.id_to_name = id_to_name_dict
        self.unique_ids = sorted(list(id_to_name_dict.keys()))
        
        self.id_to_idx = {adr_id: i for i, adr_id in enumerate(self.unique_ids)}
        self.idx_to_id = {i: adr_id for i, adr_id in enumerate(self.unique_ids)}
        
        self.vocab_size = len(self.unique_ids)

    def encode(self, adr_list):
        """
        Takes a list of ADR IDs and returns a binary vector (1s and 0s).
        Example: ['10028553', '10003041'] -> [0, 1, 0, 0, 1...]
        """
        vector = np.zeros(self.vocab_size, dtype=np.int8)
        
        for adr_id in adr_list:
            if adr_id in self.id_to_idx:
                idx = self.id_to_idx[adr_id]
                vector[idx] = 1
            else:
                print(f"Warning: ADR ID {adr_id} not in vocabulary.")
                
        return vector

    def decode(self, vector):
        """
        Takes a binary vector and returns a list of human-readable ADR names.
        """
        decoded_names = []
        
        active_indices = np.where(vector == 1)[0]
        
        for idx in active_indices:
            adr_id = self.idx_to_id[idx]
            name = self.id_to_name.get(adr_id, "Unknown ADR")
            decoded_names.append(name)
            
        return decoded_names

    def decode_indices(self, indices):
        """
        Takes a list or array of indices (e.g., [42, 105, 300]) 
        and returns the corresponding ADR names.
        """
        return [self.id_to_name.get(self.idx_to_id[idx], "Unknown ADR") for idx in indices]

    def decode_top_k(self, confidence_array, k=5):
        """
        Takes the raw probability array from the model, finds the top K 
        highest values, and returns names + their confidence scores.
        """
        # Get indices of the top k probabilities
        top_indices = np.argsort(confidence_array)[-k:][::-1]
        
        results = []
        for idx in top_indices:
            adr_id = self.idx_to_id[idx]
            name = self.id_to_name.get(adr_id, "Unknown ADR")
            conf = confidence_array[idx]
            results.append({"name": name, "confidence": round(float(conf), 4)})
            
        return results

    def decode_with_threshold(self, confidence_array, threshold=0.5):
        """
        Returns all ADRs that pass a specific confidence threshold.
        Useful for seeing everything the model is "sure" about.
        """
        active_indices = np.where(confidence_array >= threshold)[0]
        
        # Sort them by confidence (highest first)
        active_indices = active_indices[np.argsort(confidence_array[active_indices])[::-1]]
        
        return [self.id_to_name.get(self.idx_to_id[idx], "Unknown ADR") for idx in active_indices]


adrdf = pd.read_parquet(config['dataset']["adr"])
id_name_dict = dict(zip(adrdf['meddra_id'], adrdf['meddra_name']))
adr_manager = ADRData(id_name_dict)

drug_to_adr_list = adrdf.groupby('rxnorm_ingredient_id')['meddra_id'].apply(list).to_dict()

def get_encoded_adr(drug_id):
    # Get the list of ADRs for this drug, or an empty list if not found
    adrs = drug_to_adr_list.get(drug_id, [])
    return adr_manager.encode(adrs)

# 2. Map the drug_id (e.g., rxcui) to the encoded vector
# This will create a column where each cell is a numpy array
df['adr'] = df['rxcui'].map(get_encoded_adr)
print(f"Total rows with ADRs: {df['adr'].apply(lambda x: x.sum() > 0).sum()}")

# print number of unique protein and drug ids

print(df["protein_id"].nunique())
print(df["drug_id"].nunique())

# print number of unique rxcui
print(df["rxcui"].nunique())
print(df.head(1))



df.info()

from sklearn.model_selection import train_test_split

# 1. Get all unique protein IDs
unique_proteins = df['protein_id'].unique()

# 2. Split protein IDs (not rows) to ensure no leakage
# We'll reserve 10% of proteins for Test and 10% for Validation
train_prot_ids, temp_prot_ids = train_test_split(
    unique_proteins, 
    test_size=0.20, 
    random_state=42
)

val_prot_ids, test_prot_ids = train_test_split(
    temp_prot_ids, 
    test_size=0.50, 
    random_state=42
)

# 3. Create the dataframes based on these ID splits
train_df = df[df['protein_id'].isin(train_prot_ids)]
val_df = df[df['protein_id'].isin(val_prot_ids)]
test_df = df[df['protein_id'].isin(test_prot_ids)]

# --- Verification & Metrics ---
print(f"--- Final Dataset Sizes ---")
print(f"Train Set: {len(train_df)} rows ({len(train_prot_ids)} proteins)")
print(f"Val Set:   {len(val_df)} rows ({len(val_prot_ids)} proteins) - [Model Selection]")
print(f"Test Set:  {len(test_df)} rows ({len(test_prot_ids)} proteins) - [Cold-Protein Eval]")

# 4. Recalculate Positive Weight for Training
num_neg = (train_df['label'] == 0).sum()
num_pos = (train_df['label'] == 1).sum()

# Avoid division by zero just in case
pos_weight_value = num_neg / num_pos if num_pos > 0 else 1.0

print(f"\nNew Positive Weight: {pos_weight_value:.2f}")
print(f"Positive/Negative Ratio in Train: 1:{num_neg/num_pos:.2f}")
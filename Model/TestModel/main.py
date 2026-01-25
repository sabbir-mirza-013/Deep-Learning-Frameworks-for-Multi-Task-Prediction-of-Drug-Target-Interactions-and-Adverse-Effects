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
print(df.head(5))


from sklearn.model_selection import train_test_split

# 1. Identify the 'Cold' Validation Proteins (The 10 unique ones)
cold_prot_ids = df['protein_id'].drop_duplicates().sample(n=10, random_state=42).values
val_cold_df = df[df['protein_id'].isin(cold_prot_ids)]

# 2. Get the remaining data (everything else)
remaining_df = df[~df['protein_id'].isin(cold_prot_ids)]

# 3. Perform a standard 80/20 split on the remaining data
# We use stratify=y to keep the label distribution consistent
train_df, test_df = train_test_split(
    remaining_df, 
    test_size=0.20, 
    random_state=42, 
    stratify=remaining_df['label']
)

print(f"--- Final Dataset Sizes ---")
print(f"Train Set: {len(train_df)} rows (Standard Training)")
print(f"Test Set:  {len(test_df)} rows (General performance)")
print(f"Val Set:   {len(val_cold_df)} rows (Cold-Protein generalization)")

# 4. Recalculate Positive Weight for Training
num_neg = (train_df['label'] == 0).sum()
num_pos = (train_df['label'] == 1).sum()
pos_weight_value = num_neg / num_pos

print(f"\nNew Positive Weight: {pos_weight_value:.2f}")



class FusionModule(nn.Module):
    def __init__(self, dim1, dim2, output_dim):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(dim1 + dim2, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    def forward(self, e1, e2):
        return self.fusion(torch.cat([e1, e2], dim=1))

class MultiTaskFusionVAE(nn.Module):
    def __init__(self, drug_dims, prot_dims, adr_dim=4817, fused_dim=768, latent_dim=256):
        super().__init__()
        
        # 1. Fusion Layers
        # Uses drug_dims (list/tuple e.g., [1024, 512]) and prot_dims
        self.drug_fusion = FusionModule(drug_dims[0], drug_dims[1], fused_dim)
        self.prot_fusion = FusionModule(prot_dims[0], prot_dims[1], fused_dim)
        
        # 2. Contextual Encoder 
        # Variable input: drug fused_dim + protein fused_dim
        encoder_input_dim = fused_dim * 2 
        self.context_encoder = nn.Sequential(
            nn.Linear(encoder_input_dim, 1024), 
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU()
        )
        
        # 3. VAE Bottleneck
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)
        
        # 4. Multi-task Heads
        self.adr_decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, adr_dim)
        )
        
        self.dti_head = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            # nn.Linear(256, 128),
            # nn.ReLU(),
            nn.Linear(256, 1)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, d1, d2, p1, p2):
        fused_drug = self.drug_fusion(d1, d2) 
        fused_prot = self.prot_fusion(p1, p2)
        
        # Concat the two fused representations
        context_input = torch.cat([fused_drug, fused_prot], dim=1)
        context = self.context_encoder(context_input)
        
        mu = self.fc_mu(context)
        logvar = self.fc_logvar(context)
        z = self.reparameterize(mu, logvar)
        
        adr_logits = self.adr_decoder(z)
        dti_logits = self.dti_head(z)
        
        return dti_logits, adr_logits, mu, logvar
    
class MultiTaskLossWrapper(nn.Module):
    def __init__(self, num_tasks=2):
        super(MultiTaskLossWrapper, self).__init__()
        # These represent the 'noise' or uncertainty of each task
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, loss_dti, loss_adr):
        # Task 1: DTI
        precision1 = torch.exp(-self.log_vars[0])
        loss1 = precision1 * loss_dti + self.log_vars[0]

        # Task 2: ADR
        precision2 = torch.exp(-self.log_vars[1])
        loss2 = precision2 * loss_adr + self.log_vars[1]

        return loss1 + loss2

from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, 
    precision_recall_curve, auc, precision_score, recall_score, average_precision_score
)

def evaluate_multitask(model, dataloader, device):
    model.eval()
    
    # Storage for DTI targets and predictions
    dti_true, dti_probs = [], []
    # Storage for ADR targets and predictions
    adr_true, adr_probs = [], []

    with torch.no_grad():
        for d1, d2, p1, p2, labels, adr_targets in dataloader:
            d1, d2, p1, p2 = d1.to(device), d2.to(device), p1.to(device), p2.to(device)
            
            # Forward pass
            dti_logits, adr_logits, _, _ = model(d1, d2, p1, p2)
            
            # Convert to probabilities
            dti_p = torch.sigmoid(dti_logits).cpu().numpy()
            adr_p = torch.sigmoid(adr_logits).cpu().numpy()
            
            dti_true.extend(labels.numpy())
            dti_probs.extend(dti_p)
            
            adr_true.extend(adr_targets.numpy())
            adr_probs.extend(adr_p)

    # --- 1. DTI METRICS ---
    dti_true = np.array(dti_true)
    dti_probs = np.array(dti_probs).flatten()
    dti_preds = (dti_probs > 0.5).astype(int)

    # Calculate Precision-Recall AUC (AUPRC)
    precision_pts, recall_pts, _ = precision_recall_curve(dti_true, dti_probs)
    auprc = auc(recall_pts, precision_pts)

    dti_results = {
        "DTI_Accuracy": accuracy_score(dti_true, dti_preds),
        "DTI_F1": f1_score(dti_true, dti_preds),
        "DTI_Precision": precision_score(dti_true, dti_preds),
        "DTI_Recall": recall_score(dti_true, dti_preds),
        "DTI_AUROC": roc_auc_score(dti_true, dti_probs),
        "DTI_AUPRC": auprc
    }

    # --- 2. ADR METRICS (Micro-averaged across 4,817 labels) ---
    adr_true = np.array(adr_true)
    adr_probs = np.array(adr_probs)
    adr_preds = (adr_probs > 0.5).astype(int)
    
    # For ADRs, we usually report Micro-average because labels are sparse
    adr_results = {
        "ADR_Micro_AUROC": roc_auc_score(adr_true, adr_probs, average='micro'),
        "ADR_Macro_AUROC": roc_auc_score(adr_true, adr_probs, average='macro'),
        "ADR_Weighted_AUROC": roc_auc_score(adr_true, adr_probs, average='weighted'),
        "ADR_Weighted_AUPRC": average_precision_score(adr_true, adr_probs, average='weighted'), # Approx
        "ADR_Micro_AUPRC": average_precision_score(adr_true, adr_probs, average='micro'), # Approx
        "ADR_Macro_AUPRC": average_precision_score(adr_true, adr_probs, average='macro'), # Approx
        "ADR_F1": f1_score(adr_true, adr_preds, average="weighted")
    }

    return {**dti_results, **adr_results}


class DTIVAE_Dataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe.reset_index(drop=True)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Drug Inputs (Fusion)
        d1 = torch.tensor(row['drug_emb_1'], dtype=torch.float)
        d2 = torch.tensor(row['drug_emb_2'], dtype=torch.float)
        
        # Protein Inputs (Fusion)
        p1 = torch.tensor(row['prot_emb_1'], dtype=torch.float)
        p2 = torch.tensor(row['prot_emb_2'], dtype=torch.float)
        
        # Targets
        label = torch.tensor(row['label'], dtype=torch.float)
        adr_vector = torch.tensor(row['adr'], dtype=torch.float)
        
        return d1, d2, p1, p2, label, adr_vector


train_loader = DataLoader(DTIVAE_Dataset(train_df), batch_size=64, shuffle=True)
val_loader = DataLoader(DTIVAE_Dataset(val_cold_df), batch_size=64, shuffle=False)
test_loader = DataLoader(DTIVAE_Dataset(test_df), batch_size=64, shuffle=False)




def train_model(model, train_loader, test_loader, val_loader, pos_weight, epochs=50, lr=1e-4, device='cuda', monitor=None):
    model.to(device)
    
    # 1. Initialize the dynamic balancer
    loss_balancer = MultiTaskLossWrapper(num_tasks=2).to(device)
    
    # 2. Setup Optimizer to include balancer parameters
    # This is critical: the balancer needs to be updated by the same optimizer
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': loss_balancer.parameters(), 'lr': lr}
    ], lr=lr, weight_decay=1e-4)
    
    # Scheduler tracks Val DTI AUPRC to know when to slow down
    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5, verbose=True)
    
    # Weighting for the binary DTI task imbalance
    dti_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
    
    best_val_auprc = 0
    
    print(f"Starting Training on {device}...")
    
    for epoch in range(epochs):
        model.train()
        train_losses = []
        
        # VAE KL-annealing: Slowly introduce KL loss to keep latent space organized
        beta = min(0.04, epoch * 0.001)
        
        for d1, d2, p1, p2, labels, adr_targets in train_loader:
            d1, d2, p1, p2 = d1.to(device), d2.to(device), p1.to(device), p2.to(device)
            labels, adr_targets = labels.to(device), adr_targets.to(device)
            
            optimizer.zero_grad()
            
            # Forward Pass
            dti_logits, adr_logits, mu, logvar = model(d1, d2, p1, p2)
            
            # Individual Task Losses
            loss_dti = dti_criterion(dti_logits.squeeze(), labels)
            loss_adr = F.binary_cross_entropy_with_logits(adr_logits, adr_targets)
            
            # VAE Loss (KL Divergence)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            kl_loss /= (adr_targets.size(0) * adr_targets.size(1))
            
            # 3. Apply Dynamic Balancing (Replaces static Alpha)
            balanced_task_loss = loss_balancer(loss_dti, loss_adr)
            total_loss = balanced_task_loss + (beta * kl_loss)

            
            total_loss.backward()
            
            # 4. Gradient Clipping: Prevents spikes that cause oscillations
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_losses.append(total_loss.item())
            
        # Evaluation step after each epoch
        test_metrics = evaluate_multitask(model, test_loader, device)
        val_metrics = evaluate_multitask(model, val_loader, device)

        current_auprc = val_metrics['DTI_AUPRC']

        is_best = current_auprc > best_val_auprc
        # 5. Checkpointing the Best Model
        if current_auprc > best_val_auprc:
            best_val_auprc = current_auprc
            torch.save(model.state_dict(), 'best_multitask_model.pth')
            print(f"*** New Best Model Saved (AUPRC: {best_val_auprc:.4f}) ***")
            
        # Update learning rate based on performance
        scheduler.step(current_auprc)
        
        # Calculate learned weights for the thesis report
        with torch.no_grad():
            w_dti = torch.exp(-loss_balancer.log_vars[0]).item()
            w_adr = torch.exp(-loss_balancer.log_vars[1]).item()
        
        table_data = [
            [
                "Test", 
                test_metrics['DTI_AUROC'], 
                test_metrics['DTI_AUPRC'], 
                test_metrics['DTI_F1'], 
                test_metrics['ADR_Micro_AUROC'],
                test_metrics['ADR_Weighted_AUROC'],
                test_metrics['ADR_Micro_AUPRC'],
                test_metrics['ADR_Weighted_AUPRC'],
                test_metrics['ADR_F1']
            ],
            [
                "Validation", 
                val_metrics['DTI_AUROC'], 
                val_metrics['DTI_AUPRC'], 
                val_metrics['DTI_F1'], 
                val_metrics['ADR_Micro_AUROC'],
                val_metrics['ADR_Weighted_AUROC'],
                val_metrics['ADR_Micro_AUPRC'],
                val_metrics['ADR_Weighted_AUPRC'],
                val_metrics['ADR_F1']
            ]
            
        ]
        headers = ["SET","DTI AUROC", "DTI AUPRC", "DTI F1", "ADR MICRO AUROC","ADR WEIGHTED AUROC", "ADR MICRO AUPRC","ADR WEIGHTED AUPRC", "ADR_F1"]

        monitor.log_epoch(epoch, headers, table_data, best = is_best)
        

        # 3. Print the report
        print(f"\n🚀 Epoch {epoch+1}/{epochs} | Loss: {np.mean(train_losses):.4f}")
        print(tabulate(table_data, headers=headers, tablefmt="fancy_grid", floatfmt=".4f"))
        print(f"Weights: w_dti={w_dti:.4f}, w_adr={w_adr:.4f}\n")

    return model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

sample_row = train_df.iloc[0]

# Detect Drug Dimensions
d1_dim = len(sample_row['drug_emb_1'])
d2_dim = len(sample_row['drug_emb_2'])

# Detect Protein Dimensions
p1_dim = len(sample_row['prot_emb_1'])
p2_dim = len(sample_row['prot_emb_2'])

# Detect ADR Dimension
num_adrs = adr_manager.vocab_size

print(f"--- Detected Dimensions ---")
print(f"Drug Inputs: {d1_dim}, {d2_dim}")
print(f"Protein Inputs: {p1_dim}, {p2_dim}")
print(f"ADR Output: {num_adrs}")

# --- 2. INITIALIZE MODEL ---
# Dim sizes depend on your specific embeddings (e.g., 768 or 1024)
model = MultiTaskFusionVAE(
    drug_dims=[d1_dim, d2_dim], 
    prot_dims=[p1_dim, p2_dim], 
    adr_dim=num_adrs, 
    fused_dim=1028,
    latent_dim=512
).to(device)



import os
import sys
sys.path.append(os.path.abspath(".."))
from training_monitor import TrainingMonitor


monitor = TrainingMonitor(server_url='http://localhost:8080', model_name='VAE_Main')

monitor.connect()


trained_model = train_model(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    val_loader=val_loader,
    pos_weight=1.84,
    epochs=50,
    lr=1e-3,
    device=device,
    monitor=monitor
)

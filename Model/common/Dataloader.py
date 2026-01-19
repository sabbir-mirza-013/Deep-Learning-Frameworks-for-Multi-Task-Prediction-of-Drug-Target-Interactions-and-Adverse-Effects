class ADRManager:
    def __init__(self, latent_dim):
        # Dictionary to store {adr_id: list_of_embeddings}
        self.adr_registry = {}

        # Final averaged prototypes {adr_id: tensor_point}
        self.prototypes = {}

    def update_registry(self, adr_ids, drug_embeddings, protein_embeddings):
        """
        Call this during an epoch to collect embeddings for each ADR.
        As per your plan: ADR = avg(Drugs + Proteins involved)
        """
        for i, adr_id in enumerate(adr_ids):
            if adr_id not in self.adr_registry:
                self.adr_registry[adr_id] = []
            
            # Combine the drug and the protein it interacted with for this ADR
            combined_context = (drug_embeddings[i] + protein_embeddings[i]) / 2
            self.adr_registry[adr_id].append(combined_context.detach())

    def compute_prototypes(self):
        """Compute the final 'point' for every ADR in the latent space"""
        for adr_id, embeddings in self.adr_registry.items():
            self.prototypes[adr_id] = torch.stack(embeddings).mean(dim=0)
            
        return self.prototypes


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


class DTIDataset(Dataset):
    """
    Custom Dataset for Drug-Target Interaction data
    """
    
    def __init__(self, dataframe, transform=None):
        self.data = dataframe
        self.transform = transform
        
        # Preprocess: Convert all features to numpy arrays
        self._preprocess_features()
        
    def _preprocess_features(self):
        """Convert features from various formats to numpy arrays"""
        print("Preprocessing features...")
        
        # Process each feature column
        feature_columns = ['gvp_embedding', 'esm_embedding', 'egnn_embedding', 'chemberta_embedding']
        
        for col in feature_columns:
            if col in self.data.columns:
                # Convert lists/arrays to numpy arrays
                self.data[col] = self.data[col].apply(
                    lambda x: np.array(x) if isinstance(x, (list, np.ndarray)) else x
                )
                
                # Check and fix dimensions
                sample_shape = self.data[col].iloc[0].shape if len(self.data) > 0 else None
                print(f"  {col}: {sample_shape}")
        
        # Process ADR labels
        if 'adr_embedding' in self.data.columns:
            self.data['adr_embedding'] = self.data['adr_embedding'].apply(
                lambda x: np.array(x, dtype=np.float32) if isinstance(x, (list, np.ndarray)) else x
            )
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        
        # Extract features
        features = {
            'protein_id': str(sample.get('target_uniprot_id', '')),
            'drug_id': str(sample.get('rxcui', '')),
            
            # Protein features
            'protein_gvp': torch.FloatTensor(sample.get('gvp_embedding', np.zeros(1024))),
            'protein_esm': torch.FloatTensor(sample.get('esm_embedding', np.zeros(1280))),
            
            # Drug features
            'drug_egnn': torch.FloatTensor(sample.get('egnn_embedding', np.zeros(256))),
            'drug_chemberta': torch.FloatTensor(sample.get('chemberta_embedding', np.zeros(384))),
            
            # Labels
            'dti_labels': torch.FloatTensor([sample.get('label', 0)]),  # DTI binary label
            'adr_labels': torch.FloatTensor(sample.get('adr_embedding', np.zeros(4817)))  # Multi-label ADRs
        }
        
        if self.transform:
            features = self.transform(features)
            
        return features



print("Loading parquet file...")
df = pd.read_parquet('../ContrastiveLearningModel/train_dti.parquet')  # Replace with your actual file path
test_df = pd.read_parquet('../ContrastiveLearningModel/test_dti.parquet')

import pandas as pd

adrdf = pd.read_parquet("../../Data/final_rxnorm_meddra_v2.parquet")
id_name_dict = dict(zip(adrdf['meddra_id'], adrdf['meddra_name']))
adr_manager = ADRData(id_name_dict)

df['adr_embedding'] = df['adr_ids'].apply(lambda x: adr_manager.encode(x))
test_df['adr_embedding'] = test_df['adr_ids'].apply(lambda x: adr_manager.encode(x))
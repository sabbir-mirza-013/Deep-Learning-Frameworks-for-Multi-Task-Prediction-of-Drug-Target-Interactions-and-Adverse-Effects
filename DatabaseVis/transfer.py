import pandas as pd
import sqlite3

# Initialize connection
conn = sqlite3.connect('main.db')

# 1. Migrate MedDRA mappings
df_meddra = pd.read_parquet('../Data/final_rxnorm_meddra_v2.parquet')
df_meddra.to_sql('meddra_mappings', conn, if_exists='replace', index=False)

# 2. Migrate Onside Common (Features/Labels)
df_scope = pd.read_parquet('../Data/scope_onside_common_v3.parquet')
df_scope.to_sql('drug_target_interactions', conn, if_exists='replace', index=False)

# Optional: Create an index on 'rxcui' or 'drug_chembl_id' for faster joins later
conn.execute('CREATE INDEX idx_rxcui ON drug_target_interactions(rxcui)')

conn.close()
print("Migration complete!")
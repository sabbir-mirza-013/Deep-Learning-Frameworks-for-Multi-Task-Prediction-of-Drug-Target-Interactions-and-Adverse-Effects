import sqlite3
import os
from typing import List, Optional, Any, Dict
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException


app = FastAPI(title="DatabaseVis Server")

# CONSTANTS
DB_PATH = "main.db"
CLIENT_DIR = "client"

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class TableStats(BaseModel):
    table_name: str
    row_count: int

class Interaction(BaseModel):
    drug_chembl_id: Optional[str]
    target_uniprot_id: Optional[str]
    label: Optional[int]
    smiles: Optional[str]
    sequence: Optional[str]
    molfile_3d: Optional[str]
    rxcui: Optional[str]

class Mapping(BaseModel):
    rxnorm_ingredient_id: Optional[str]
    meddra_id: Optional[int]
    meddra_name: Optional[str]

class PaginatedResponse(BaseModel):
    data: List[Any]
    total: int
    page: int
    limit: int

# Database Helper
def get_db_connection():
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        raise HTTPException(status_code=500, detail="Database connection failed")

# Endpoints
@app.get("/api/stats", response_model=List[TableStats])
async def get_stats():
    conn = get_db_connection()
    cursor = conn.cursor()
    stats = []
    
    tables = [("drug_target_interactions",), ("meddra_mappings",)]
    
    for table in tables:
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {table[0]}")
            count = cursor.fetchone()[0]
            stats.append(TableStats(table_name=table[0], row_count=count))
        except Exception as e:
            print(f"Error counting {table[0]}: {e}")
            
    conn.close()
    return stats

@app.get("/api/detailed-stats")
async def get_detailed_stats():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    stats = {}
    
    try:
        # Basic Counts
        cursor.execute("SELECT COUNT(*) FROM drug_target_interactions")
        stats['total_interactions'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM meddra_mappings")
        stats['total_mappings'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT drug_chembl_id) FROM drug_target_interactions")
        stats['unique_drugs'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT target_uniprot_id) FROM drug_target_interactions")
        stats['unique_proteins'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT meddra_id) FROM meddra_mappings")
        stats['unique_meddra'] = cursor.fetchone()[0]
        
        # Label Distribution
        cursor.execute("SELECT label, COUNT(*) as count FROM drug_target_interactions GROUP BY label")
        labels = {str(row['label']): row['count'] for row in cursor.fetchall()}
        stats['label_distribution'] = labels
        
    except Exception as e:
        print(f"Error fetching detailed stats: {e}")
        conn.close()
        raise HTTPException(status_code=500, detail="Failed to fetch statistics")
    
    conn.close()
    return stats

@app.get("/api/top-drugs")
async def get_top_drugs(limit: int = 10):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT drug_chembl_id, COUNT(*) as count 
            FROM drug_target_interactions 
            GROUP BY drug_chembl_id 
            ORDER BY count DESC 
            LIMIT ?
        """, (limit,))
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()

@app.get("/api/top-targets")
async def get_top_targets(limit: int = 10):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT target_uniprot_id, COUNT(*) as count 
            FROM drug_target_interactions 
            GROUP BY target_uniprot_id 
            ORDER BY count DESC 
            LIMIT ?
        """, (limit,))
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()

@app.get("/api/top-side-effects")
async def get_top_side_effects(limit: int = 10):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT meddra_name, COUNT(*) as count 
            FROM meddra_mappings 
            GROUP BY meddra_name 
            ORDER BY count DESC 
            LIMIT ?
        """, (limit,))
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()

@app.get("/api/interactions", response_model=PaginatedResponse)
async def get_interactions(
    page: int = 1, 
    limit: int = 20, 
    search: Optional[str] = None
):
    offset = (page - 1) * limit
    conn = get_db_connection()
    cursor = conn.cursor()
    
    query = "SELECT * FROM drug_target_interactions"
    params = []
    
    if search:
        query += " WHERE drug_chembl_id LIKE ? OR target_uniprot_id LIKE ? OR smiles LIKE ?"
        search_term = f"%{search}%"
        params.extend([search_term, search_term, search_term])
        
    # Count total
    count_query = f"SELECT COUNT(*) FROM ({query})"
    cursor.execute(count_query, params)
    total = cursor.fetchone()[0]
    
    # Fetch data
    query += f" LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    
    cursor.execute(query, params)
    rows = cursor.fetchall()
    
    data = [dict(row) for row in rows]
    conn.close()
    
    return PaginatedResponse(data=data, total=total, page=page, limit=limit)

@app.get("/api/mappings", response_model=PaginatedResponse)
async def get_mappings(
    page: int = 1, 
    limit: int = 20, 
    search: Optional[str] = None
):
    offset = (page - 1) * limit
    conn = get_db_connection()
    cursor = conn.cursor()
    
    query = "SELECT * FROM meddra_mappings"
    params = []
    
    if search:
        query += " WHERE meddra_name LIKE ? OR rxnorm_ingredient_id LIKE ?"
        search_term = f"%{search}%"
        params.extend([search_term, search_term])
        
    # Count total
    count_query = f"SELECT COUNT(*) FROM ({query})"
    cursor.execute(count_query, params)
    total = cursor.fetchone()[0]
    
    # Fetch data
    query += f" LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    
    cursor.execute(query, params)
    rows = cursor.fetchall()
    
    data = [dict(row) for row in rows]
    conn.close()
    
    return PaginatedResponse(data=data, total=total, page=page, limit=limit)

@app.get("/api/ingredient/{rxcui}")
async def get_ingredient_details(rxcui: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get Side Effects
    cursor.execute("SELECT * FROM meddra_mappings WHERE rxnorm_ingredient_id = ?", (rxcui,))
    side_effects = [dict(row) for row in cursor.fetchall()]
    
    # Get Interactions
    cursor.execute("SELECT * FROM drug_target_interactions WHERE rxcui = ?", (rxcui,))
    all_interactions = [dict(row) for row in cursor.fetchall()]
    
    conn.close()
    
    positive_interactions = [i for i in all_interactions if i['label'] == 1]
    negative_interactions = [i for i in all_interactions if i['label'] == 0]
    
    return {
        "rxcui": rxcui,
        "side_effects": side_effects,
        "positive_interactions": positive_interactions,
        "negative_interactions": negative_interactions
    }

@app.get("/api/interaction/{drug_id}/{target_id}")
async def get_interaction_details(drug_id: str, target_id: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get Interaction Data
    cursor.execute(
        "SELECT * FROM drug_target_interactions WHERE drug_chembl_id = ? AND target_uniprot_id = ?", 
        (drug_id, target_id)
    )
    row = cursor.fetchone()
    
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Interaction not found")
        
    interaction = dict(row)
    rxcui = interaction.get('rxcui')
    
    # Get Associated Side Effects
    side_effects = []
    if rxcui:
        cursor.execute("SELECT * FROM meddra_mappings WHERE rxnorm_ingredient_id = ?", (rxcui,))
        side_effects = [dict(row) for row in cursor.fetchall()]
        
    conn.close()
    
    return {
        "interaction": interaction,
        "side_effects": side_effects
    }

# Mount static files first (for assets like CSS, JS, fonts)
app.mount("/assets", StaticFiles(directory=os.path.join(CLIENT_DIR, "assets")), name="assets")

# SPA Routing - Catch all non-API routes and serve index.html


@app.exception_handler(StarletteHTTPException)
async def spa_exception_handler(request, exc):
    if exc.status_code == 404:
        # Check if the request is trying to hit an API or a specific asset
        # We don't want to serve index.html for a broken API call
        if not request.url.path.startswith("/api") and "." not in request.url.path.split("/")[-1]:
            return FileResponse(os.path.join(CLIENT_DIR, "index.html"))
    return await http_exception_handler(request, exc)

# 4. Simple Root Route
@app.get("/")
async def serve_index():
    return FileResponse(os.path.join(CLIENT_DIR, "index.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8080, reload=True)

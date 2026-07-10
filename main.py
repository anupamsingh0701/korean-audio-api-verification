import os
import re
import io
import base64
import logging
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import httpx
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("korean-audio-api")

app = FastAPI(
    title="Korean Audio Dataset Verification API",
    description="Decodes base64 audio, transcribes and parses it via AIPipe, and computes statistics.",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AudioRequest(BaseModel):
    audio_id: str = Field(..., description="ID of the audio query")
    audio_base64: str = Field(..., description="Base64-encoded audio string")

# Global dict to store the last extraction details for debugging
last_debug_info = {}

def detect_mime_type(audio_bytes: bytes) -> str:
    """Detects audio mime type from bytes header."""
    if audio_bytes.startswith(b"RIFF") and b"WAVE" in audio_bytes[:15]:
        return "audio/wav"
    elif audio_bytes.startswith(b"ID3") or audio_bytes.startswith(b"\xff\xfb") or audio_bytes.startswith(b"\xff\xf3"):
        return "audio/mp3"
    elif audio_bytes.startswith(b"FLAC"):
        return "audio/flac"
    elif audio_bytes.startswith(b"OggS"):
        return "audio/ogg"
    elif b"ftyp" in audio_bytes[4:12]:
        return "audio/mp4"
    # Fallback to audio/wav
    return "audio/wav"

async def get_aipipe_csv_extraction(audio_base64: str, mime_type: str, ext: str) -> str:
    """Uses AIPipe OpenRouter proxy with Gemini 1.5 Pro for reliable audio transcription."""
    api_key = os.environ.get("AIPIPE_TOKEN")
    if not api_key:
        raise ValueError("AIPIPE_TOKEN is not set in environment.")
        
    url = "https://aipipe.org/openrouter/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    prompt = (
        "The following audio contains speech (in Korean) reading a tabular dataset or describing table data. "
        "Please transcribe the audio, identify the table structure, and return the data as a clean CSV table. "
        "Rules:\n"
        "1. Return ONLY the raw CSV text. Do not include markdown code block formatting like ```csv or any other text.\n"
        "2. Make sure the headers represent the columns read.\n"
        "3. Ensure that all rows are correctly extracted.\n"
        "4. If numeric values are read, ensure they are written as plain numbers (no commas or units)."
    )
    
    payload = {
        "model": "google/gemini-1.5-pro",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{audio_base64}"
                        }
                    }
                ]
            }
        ],
        "temperature": 0.0
    }
    
    logger.info("Attempting extraction via AIPipe OpenRouter proxy (google/gemini-1.5-pro)...")
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            csv_text = response.json()["choices"][0]["message"]["content"].strip()
            if csv_text.startswith("```"):
                csv_text = re.sub(r"^```(?:csv)?\n|```$", "", csv_text, flags=re.MULTILINE).strip()
            logger.info("AIPipe OpenRouter Gemini CSV extraction complete.")
            return csv_text
        else:
            raise RuntimeError(f"OpenRouter proxy returned status {response.status_code}: {response.text}")

def compute_dataframe_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Computes all required statistics on a pandas DataFrame."""
    rows = int(df.shape[0])
    columns = list(df.columns)
    
    mean_dict = {}
    std_dict = {}
    var_dict = {}
    min_dict = {}
    max_dict = {}
    median_dict = {}
    mode_dict = {}
    range_dict = {}
    allowed_values = {}
    value_range = {}
    
    for col in df.columns:
        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        
        # Mode can apply to all types (first mode value)
        modes = df[col].mode()
        if not modes.empty:
            val_mode = modes.iloc[0]
            if pd.notnull(val_mode):
                mode_dict[col] = float(val_mode) if is_numeric else str(val_mode)
                
        if is_numeric:
            # Min
            val_min = df[col].min()
            if pd.notnull(val_min):
                min_dict[col] = float(val_min)
                
            # Max
            val_max = df[col].max()
            if pd.notnull(val_max):
                max_dict[col] = float(val_max)
                
            val_mean = df[col].mean()
            if pd.notnull(val_mean):
                mean_dict[col] = float(val_mean)
                
            val_std = df[col].std()
            if pd.notnull(val_std):
                std_dict[col] = float(val_std)
                
            val_var = df[col].var()
            if pd.notnull(val_var):
                var_dict[col] = float(val_var)
                
            val_median = df[col].median()
            if pd.notnull(val_median):
                median_dict[col] = float(val_median)
                
            if pd.notnull(val_min) and pd.notnull(val_max):
                range_dict[col] = float(val_max - val_min)
                value_range[col] = [float(val_min), float(val_max)]
        else:
            # Allowed unique values (exclude NaN) - only for non-numeric/categorical columns
            unique_vals = df[col].dropna().unique().tolist()
            if len(unique_vals) > 0:
                unique_vals = [v.item() if hasattr(v, "item") else v for v in unique_vals]
                try:
                    unique_vals.sort()
                except Exception:
                    pass
                allowed_values[col] = unique_vals
                
    # Correlation list format: [{"x": "col1", "y": "col2", "type": "positive"}]
    correlation_list = []
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) >= 2:
        corr_df = df[numeric_cols].corr(method="pearson")
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                c1 = numeric_cols[i]
                c2 = numeric_cols[j]
                coef = corr_df.loc[c1, c2]
                if pd.notnull(coef):
                    if coef >= 0.3:
                        ctype = "positive"
                    elif coef <= -0.3:
                        ctype = "negative"
                    else:
                        ctype = "none"
                    correlation_list.append({
                        "x": c1,
                        "y": c2,
                        "type": ctype
                    })
    
    return {
        "rows": rows,
        "columns": columns,
        "mean": mean_dict,
        "std": std_dict,
        "variance": var_dict,
        "min": min_dict,
        "max": max_dict,
        "median": median_dict,
        "mode": mode_dict,
        "range": range_dict,
        "allowed_values": allowed_values,
        "value_range": value_range,
        "correlation": correlation_list
    }

@app.post("/verify")
@app.post("/")
async def verify_audio(req: AudioRequest):
    global last_debug_info
    logger.info(f"Received request for audio_id: {req.audio_id}")
    
    # Strip any possible data uri prefix from base64
    base64_data = req.audio_base64.strip()
    if base64_data.startswith("data:"):
        parts = base64_data.split(";base64,")
        if len(parts) == 2:
            base64_data = parts[1]
            
    # Decode audio bytes
    try:
        audio_bytes = base64.b64decode(base64_data)
    except Exception as e:
        logger.error(f"Failed to decode base64: {e}")
        raise HTTPException(status_code=400, detail="Invalid base64 encoding.")
        
    mime_type = detect_mime_type(audio_bytes)
    ext = mime_type.split("/")[-1]
    logger.info(f"Detected MIME type: {mime_type}, ext: {ext}")
    
    if not os.environ.get("AIPIPE_TOKEN"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="AIPIPE_TOKEN is not configured on the server."
        )
        
    try:
        csv_text = await get_aipipe_csv_extraction(base64_data, mime_type, ext)
    except Exception as e:
        logger.error(f"AIPipe extraction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"AIPipe transcription and CSV extraction failed: {str(e)}"
        )
        
    if not csv_text:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="AIPipe returned empty CSV data."
        )
        
    logger.info(f"Extracted CSV:\n{csv_text}")
    
    # Load to DataFrame
    try:
        df = pd.read_csv(io.StringIO(csv_text))
        # Strip string values and clean column headers
        df.columns = [c.strip() for c in df.columns]
        for col in df.columns:
            if df[col].dtype == object:
                # Strip spaces
                df[col] = df[col].astype(str).str.strip()
                
                # Robust numeric cleanup: try converting to numeric after stripping common units/commas
                cleaned = df[col].str.replace(r'[\s,%\$]|cm|kg|m', '', regex=True, case=False)
                coerced = pd.to_numeric(cleaned, errors='coerce')
                
                # Coerce if at least 50% of values convert successfully (to handle notes or missing values)
                non_null_orig = df[col].dropna().shape[0]
                non_null_coerced = coerced.dropna().shape[0]
                if non_null_orig > 0 and (non_null_coerced / non_null_orig) >= 0.5:
                    df[col] = coerced
    except Exception as e:
        logger.error(f"Failed to parse CSV text into DataFrame: {e}")
        raise HTTPException(status_code=500, detail="Extracted CSV format is invalid.")
        
    # Store extraction details for debugging
    last_debug_info = {
        "audio_id": req.audio_id,
        "mime_type": mime_type,
        "csv_text": csv_text,
        "columns": list(df.columns),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "df_json": df.to_dict(orient="records")
    }
        
    # Compute stats
    stats = compute_dataframe_statistics(df)
    logger.info(f"Successfully computed statistics for {req.audio_id}: {stats}")
    return stats

@app.get("/debug")
async def get_debug():
    return last_debug_info

@app.get("/")
async def root():
    return {"status": "ok", "message": "Korean Audio API is running"}

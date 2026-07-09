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
    description="Decodes base64 audio, transcribes it using Whisper/Gemini/AIPipe, parses tabular data, and computes statistics.",
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

async def transcribe_audio_via_aipipe(audio_bytes: bytes, filename: str, mime_type: str) -> str:
    """Uses AIPipe proxy to transcribe the audio file via OpenAI Whisper API."""
    api_key = os.environ.get("AIPIPE_TOKEN")
    if not api_key:
        raise ValueError("AIPIPE_TOKEN is not set in environment.")
        
    base_url = os.environ.get("AIPIPE_BASE_URL", "https://aipipe.org/openai/v1").rstrip("/")
    url = f"{base_url}/audio/transcriptions"
    
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    files = {
        "file": (filename, audio_bytes, mime_type)
    }
    data = {
        "model": "whisper-1"
    }
    
    logger.info(f"Transcribing audio via AIPipe at {url} (whisper-1)...")
    async with httpx.AsyncClient(timeout=45.0) as client:
        response = await client.post(url, headers=headers, files=files, data=data)
        if response.status_code == 200:
            result_text = response.json().get("text", "").strip()
            logger.info("AIPipe transcription complete.")
            return result_text
        else:
            raise RuntimeError(f"AIPipe transcription returned status {response.status_code}: {response.text}")

async def convert_transcript_to_csv_via_aipipe(transcript: str) -> str:
    """Uses AIPipe proxy to convert text transcript to a CSV table using GPT-4o-mini."""
    api_key = os.environ.get("AIPIPE_TOKEN")
    if not api_key:
        raise ValueError("AIPIPE_TOKEN is not set in environment.")
        
    base_url = os.environ.get("AIPIPE_BASE_URL", "https://aipipe.org/openai/v1").rstrip("/")
    url = f"{base_url}/chat/completions"
    
    model_name = os.environ.get("AIPIPE_MODEL", "gpt-4o-mini")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    prompt = (
        f"The text below is a transcript of an audio recording of a dataset/table being read:\n"
        f"\"{transcript}\"\n\n"
        f"Please convert this transcript into a clean CSV table with a header.\n"
        f"Rules:\n"
        f"1. Return ONLY the raw CSV text. Do not include markdown code block formatting (like ```csv) or other explanations.\n"
        f"2. Ensure that numbers contain no formatting (no commas, no units)."
    )
    
    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0
    }
    
    logger.info(f"Converting transcript to CSV via AIPipe at {url} ({model_name})...")
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            csv_text = response.json()["choices"][0]["message"]["content"].strip()
            if csv_text.startswith("```"):
                csv_text = re.sub(r"^```(?:csv)?\n|```$", "", csv_text, flags=re.MULTILINE).strip()
            logger.info("AIPipe CSV conversion complete.")
            return csv_text
        else:
            raise RuntimeError(f"AIPipe CSV conversion returned status {response.status_code}: {response.text}")

async def get_gemini_csv_extraction(audio_base64: str, mime_type: str) -> str:
    """Uses Gemini API to transcribe and directly extract CSV data from audio."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set in environment.")
        
    model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    
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
        "contents": [
            {
                "parts": [
                    {
                        "inlineData": {
                            "mimeType": mime_type,
                            "data": audio_base64
                        }
                    },
                    {
                        "text": prompt
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 2048
        }
    }
    
    logger.info(f"Sending audio directly to Gemini {model_name} for CSV extraction...")
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(gemini_url, json=payload)
        if response.status_code == 200:
            result = response.json()
            csv_text = result["contents"][0]["parts"][0]["text"].strip()
            # Strip markdown if Gemini included it
            if csv_text.startswith("```"):
                csv_text = re.sub(r"^```(?:csv)?\n|```$", "", csv_text, flags=re.MULTILINE).strip()
            logger.info("Successfully extracted CSV using Gemini.")
            return csv_text
        else:
            raise RuntimeError(f"Gemini API returned status {response.status_code}: {response.text}")

def transcribe_local_whisper(audio_path: str) -> str:
    """Uses a local Whisper model to transcribe the audio file as fallback."""
    logger.info("Loading local Whisper pipeline...")
    from transformers import pipeline
    # Use openai/whisper-tiny for speed and low CPU/RAM consumption
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")
    logger.info(f"Transcribing {audio_path} using local Whisper...")
    result = asr(audio_path)
    logger.info("Local Whisper transcription complete.")
    text = result.get("text", "").strip()
    if not text:
        raise ValueError("Local Whisper returned empty transcript.")
    return text

async def convert_transcript_to_csv_via_gemini(transcript: str) -> str:
    """Uses Gemini API to convert text transcript to a CSV table."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set in environment.")
        
    model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    
    prompt = (
        f"The following text is a transcript of an audio recording of a dataset being read:\n"
        f"\"{transcript}\"\n\n"
        f"Please convert this transcript into a clean CSV table with a header.\n"
        f"Return ONLY the raw CSV text. Do not include markdown code block formatting or other explanations."
    )
    
    payload = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 2048
        }
    }
    
    logger.info("Converting transcript to CSV using Gemini...")
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(gemini_url, json=payload)
        if response.status_code == 200:
            result = response.json()
            csv_text = result["contents"][0]["parts"][0]["text"].strip()
            if csv_text.startswith("```"):
                csv_text = re.sub(r"^```(?:csv)?\n|```$", "", csv_text, flags=re.MULTILINE).strip()
            return csv_text
        else:
            raise RuntimeError(f"Gemini CSV conversion returned status {response.status_code}: {response.text}")

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
        # Check if numeric
        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        
        # Allowed unique values (exclude NaN)
        unique_vals = df[col].dropna().unique().tolist()
        # Convert numpy types to python native types
        unique_vals = [v.item() if hasattr(v, "item") else v for v in unique_vals]
        try:
            unique_vals.sort()
        except Exception:
            pass
        allowed_values[col] = unique_vals
        
        # Min
        val_min = df[col].min()
        if pd.notnull(val_min):
            min_dict[col] = float(val_min) if is_numeric else str(val_min)
            
        # Max
        val_max = df[col].max()
        if pd.notnull(val_max):
            max_dict[col] = float(val_max) if is_numeric else str(val_max)
            
        # Mode
        modes = df[col].mode()
        if not modes.empty:
            val_mode = modes.iloc[0]
            if pd.notnull(val_mode):
                mode_dict[col] = float(val_mode) if is_numeric else str(val_mode)
                
        # Numeric-only stats
        if is_numeric:
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
                
    # Correlation matrix of numeric columns
    corr_df = df.corr(numeric_only=True)
    # Replace NaN with None for JSON compliance
    corr_df = corr_df.where(pd.notnull(corr_df), None)
    correlation_list = corr_df.values.tolist() if not corr_df.empty else []
    
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
    
    csv_text = None
    errors = []
    
    # Method 1: Try AIPipe if token is available
    if os.environ.get("AIPIPE_TOKEN"):
        logger.info("Attempting transcription via AIPipe...")
        filename = f"audio_{req.audio_id}.{ext}"
        try:
            transcript = await transcribe_audio_via_aipipe(audio_bytes, filename, mime_type)
            if transcript:
                logger.info(f"AIPipe transcript: {transcript}")
                csv_text = await convert_transcript_to_csv_via_aipipe(transcript)
                if not csv_text:
                    errors.append("AIPipe: Transcript was generated, but CSV parsing returned empty.")
            else:
                errors.append("AIPipe: Transcript returned empty.")
        except Exception as e:
            logger.error(f"AIPipe method failed: {e}")
            errors.append(f"AIPipe method failed: {str(e)}")
    else:
        errors.append("AIPipe: skipped (AIPIPE_TOKEN not set in env).")
            
    # Method 2: Try Gemini Direct Extraction (fallback if AIPipe fails or isn't set)
    if not csv_text:
        if os.environ.get("GEMINI_API_KEY"):
            logger.info("Attempting direct extraction via Gemini...")
            try:
                csv_text = await get_gemini_csv_extraction(base64_data, mime_type)
                if not csv_text:
                    errors.append("Gemini: Direct CSV extraction returned empty.")
            except Exception as e:
                logger.error(f"Gemini direct method failed: {e}")
                errors.append(f"Gemini direct method failed: {str(e)}")
        else:
            errors.append("Gemini direct: skipped (GEMINI_API_KEY not set in env).")
        
    # Method 3: Try local Whisper fallback
    if not csv_text:
        logger.info("Attempting local Whisper fallback...")
        temp_filename = f"temp_{req.audio_id}.{ext}"
        try:
            with open(temp_filename, "wb") as f:
                f.write(audio_bytes)
            transcript = transcribe_local_whisper(temp_filename)
            if transcript:
                logger.info(f"Local transcript: {transcript}")
                csv_text = await convert_transcript_to_csv_via_gemini(transcript)
                if not csv_text:
                    errors.append("Whisper Fallback: Transcribed text generated, but Gemini CSV conversion failed.")
            else:
                errors.append("Whisper Fallback: Transcription returned empty.")
        except Exception as e:
            logger.error(f"Local Whisper fallback failed: {e}")
            errors.append(f"Whisper Fallback failed: {str(e)}")
        finally:
            if os.path.exists(temp_filename):
                try:
                    os.remove(temp_filename)
                except Exception:
                    pass
                    
    if not csv_text:
        detailed_error = " | ".join(errors)
        logger.error(f"Verification failed completely. Logs: {detailed_error}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Transcription / CSV extraction failed. Detailed logs: {detailed_error}"
        )
        
    logger.info(f"Extracted CSV:\n{csv_text}")
    
    # Load to DataFrame
    try:
        df = pd.read_csv(io.StringIO(csv_text))
        # Strip string values and clean column headers
        df.columns = [c.strip() for c in df.columns]
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.strip()
    except Exception as e:
        logger.error(f"Failed to parse CSV text into DataFrame: {e}")
        raise HTTPException(status_code=500, detail="Extracted CSV format is invalid.")
        
    # Compute stats
    stats = compute_dataframe_statistics(df)
    logger.info(f"Successfully computed statistics for {req.audio_id}: {stats}")
    return stats

@app.get("/")
async def root():
    return {"status": "ok", "message": "Korean Audio API is running"}

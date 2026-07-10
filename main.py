import os
import io
import base64
import json as json_lib
import tempfile
import asyncio
import logging
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException, Request, status
from pydantic import BaseModel, Field
import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("korean-audio-api")

app = FastAPI(
    title="Korean Audio Dataset Verification API",
    description="Decodes base64 audio, transcribes via Groq Whisper, and parses constraints via Groq Llama 3.",
    version="1.0.0"
)

class AudioRequest(BaseModel):
    audio_id: str = Field(..., description="ID of the audio query")
    audio_base64: str = Field(..., description="Base64-encoded audio string")

# Global dict to store the last extraction details for debugging
last_debug_info = {}

async def _transcribe_async(audio_bytes: bytes, api_key: str) -> str:
    """Run Groq Whisper asynchronously using httpx."""
    url = "https://api.groq.com/openai/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {api_key}"}
    
    files = {
        "file": ("audio.wav", audio_bytes, "audio/wav")
    }
    data = {
        "model": "whisper-large-v3-turbo",
        "response_format": "text",
        "language": "ko"
    }
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(url, headers=headers, files=files, data=data)
        if response.status_code != 200:
            raise RuntimeError(f"Groq Whisper returned {response.status_code}: {response.text}")
        return response.text

async def _extract_json_async(transcript: str, api_key: str) -> dict:
    """Run Groq Llama 3.3 asynchronously using httpx."""
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    sys_prompt = '''You are a highly precise audio instruction parser.
The user provides a Korean transcript of an audio command requesting a mock dataset.
Your job is to extract the EXACT constraints requested into a strict JSON format.

RULES:
1. `rows`: Integer count of rows (e.g. 100).
2. `columns`: Array of strings for ANY AND ALL variables or features mentioned (e.g. if '키' and '몸무게' are mentioned, output ["키", "몸무게"], if '성별' is mentioned output ["성별"]). This is required if any variables are mentioned.
   **CRITICAL: Remove all spaces from column names (e.g., '점수 1' MUST be output as '점수1', '성 별' MUST be output as '성별').**
3. For all other fields (mean, std, variance, min, max, median, mode, range, allowed_values, value_range, correlation): 
   ONLY populate them if they are EXPLICITLY mentioned in the text! Otherwise, leave them as empty objects `{}` or empty arrays `[]`.
4. `allowed_values`: Map column name to array of allowed values. (e.g. if '성별' only allows '남' and '여', output {"성별": ["남", "여"]}).
5. `correlation`: Array of objects. If "키" and "몸무게" have "양의 상관관계" (positive correlation), output [{"x": "키", "y": "몸무게", "type": "positive"}]. If negative, type is "negative".

OUTPUT EXACTLY THIS JSON SCHEMA:
{
  "rows": 0,
  "columns": [],
  "mean": {},
  "std": {},
  "variance": {},
  "min": {},
  "max": {},
  "median": {},
  "mode": {},
  "range": {},
  "allowed_values": {},
  "value_range": {},
  "correlation": []
}'''

    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": transcript}
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.0
    }
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(url, headers=headers, json=payload)
        if response.status_code != 200:
            raise RuntimeError(f"Groq Llama returned {response.status_code}: {response.text}")
            
        content = response.json()["choices"][0]["message"]["content"]
        
        parsed_json = json_lib.loads(content)
        
        # Safety enforcement script to aggressively strip spaces from keys/column arrays
        if "columns" in parsed_json:
            parsed_json["columns"] = [col.replace(" ", "") for col in parsed_json["columns"]]
            
        for key in ["mean", "std", "variance", "min", "max", "median", "mode", "range", "allowed_values", "value_range"]:
            if key in parsed_json and isinstance(parsed_json[key], dict):
                cleaned_dict = {k.replace(" ", ""): v for k, v in parsed_json[key].items()}
                parsed_json[key] = cleaned_dict
                
        if "correlation" in parsed_json and isinstance(parsed_json["correlation"], list):
            for i in range(len(parsed_json["correlation"])):
                if "x" in parsed_json["correlation"][i]:
                    parsed_json["correlation"][i]["x"] = parsed_json["correlation"][i]["x"].replace(" ", "")
                if "y" in parsed_json["correlation"][i]:
                    parsed_json["correlation"][i]["y"] = parsed_json["correlation"][i]["y"].replace(" ", "")
                    
        return parsed_json

@app.post("/verify")
@app.post("/")
async def verify_audio(req: AudioRequest):
    global last_debug_info
    
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise HTTPException(
            status_code=500, 
            detail="GROQ_API_KEY environment variable is missing on the server. Please add it to Render Environment Variables."
        )
        
    last_debug_info = {"audio_id": req.audio_id, "status": "processing"}
    logger.info(f"Received request for audio_id: {req.audio_id}. Starting Groq pipeline.")
    
    # 1. Decode
    try:
        # Strip any possible data uri prefix
        base64_data = req.audio_base64.strip()
        if base64_data.startswith("data:"):
            parts = base64_data.split(";base64,")
            if len(parts) == 2:
                base64_data = parts[1]
                
        audio_bytes = base64.b64decode(base64_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid audio_base64: {e}")

    # 2. Transcribe with Whisper
    try:
        transcript = await _transcribe_async(audio_bytes, groq_api_key)
        last_debug_info["transcript"] = transcript
        logger.info(f"Groq Whisper Transcript: {transcript}")
    except Exception as e:
        logger.error(f"Groq Whisper error: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription error: {str(e)}")

    # 3. Extract JSON directly with Llama
    try:
        parsed_json = await _extract_json_async(transcript, groq_api_key)
        last_debug_info["parsed_json"] = parsed_json
        logger.info(f"Groq Llama Output: {parsed_json}")
        return parsed_json
    except Exception as e:
        logger.error(f"Groq Llama error: {e}")
        raise HTTPException(status_code=500, detail=f"Llama parsing error: {str(e)}")

@app.get("/debug")
async def get_debug():
    return last_debug_info

@app.get("/")
async def root():
    return {"status": "ok", "message": "Korean Audio API is running on Groq"}

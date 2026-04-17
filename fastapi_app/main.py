from fastapi import FastAPI, HTTPException
import json
import pandas as pd
import numpy as np
import os
import re
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

from src.schemas import PredictionRequest, PredictionResponse, ExtractedFeatures
from src.model_loader import load_model
from fastapi_app.prompts import EXTRACTION_PROMPT, INTERPRETATION_PROMPT

# ------------------------------------------------------------------
# Load environment variables from the project root .env file
# ------------------------------------------------------------------
env_path = Path(__file__).parent.parent / '.env'
print(f"Looking for .env at: {env_path}")
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    print(".env file loaded.")
else:
    print("WARNING: .env file not found!")

# ------------------------------------------------------------------
# Configure Groq client
# ------------------------------------------------------------------
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found in environment")

client = Groq(api_key=api_key)
MODEL = "llama-3.3-70b-versatile"

# ------------------------------------------------------------------
# Load pre-trained ML model
# ------------------------------------------------------------------
ml_model = load_model()

# ------------------------------------------------------------------
# Summary statistics from training (adjust to your own dataset)
# ------------------------------------------------------------------
TRAIN_STATS = {
    "median_price": 163000.0,
    "price_std": 80000.0,
}

app = FastAPI(title="AI Real Estate Agent")

# ------------------------------------------------------------------
# Helper: call Groq with debug logging
# ------------------------------------------------------------------
def call_llm(prompt: str, temperature: float = 0.0) -> str:
    """Call Groq API and return response text."""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        content = response.choices[0].message.content
        print(f"=== LLM RAW RESPONSE ===\n{content}\n=== END RAW RESPONSE ===")
        return content.strip() if content else ""
    except Exception as e:
        print(f"=== GROQ API ERROR ===\n{str(e)}")
        raise

# ------------------------------------------------------------------
# Prediction endpoint
# ------------------------------------------------------------------
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    query = request.query

    # -------------------- Stage 1: Feature Extraction --------------------
    try:
        extraction_prompt = EXTRACTION_PROMPT.format(query=query)
        content = call_llm(extraction_prompt, temperature=0.0)

        if not content:
            raise ValueError("Groq returned an empty response")

        # Extract the JSON object from the response (robust against markdown fences)
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            # Fallback: remove markdown fences manually
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            json_str = content.strip()

        extracted_dict = json.loads(json_str)
        extracted = ExtractedFeatures(**extracted_dict)

    except Exception as e:
        print("=== LLM EXTRACTION ERROR ===")
        print(str(e))
        raise HTTPException(status_code=422, detail=f"LLM extraction failed: {str(e)}")

    # -------------------- Stage 2: ML Prediction --------------------
        # Prepare input for ML model
    input_data = extracted.dict(by_alias=True, exclude={'completeness', 'missing_features'})
    input_df = pd.DataFrame([input_data])
    
    # Replace Python None with np.nan so SimpleImputer can fill missing values
    input_df = input_df.replace({None: np.nan})
    
    expected_cols = ml_model.feature_names_in_
    input_df = input_df[expected_cols]

    pred_log = ml_model.predict(input_df)[0]
    pred_price = float(np.expm1(pred_log))   # Convert from log1p scale

    # -------------------- Stage 3: Interpretation --------------------
    try:
        interpretation_prompt = INTERPRETATION_PROMPT.format(
            query=query,
            price=pred_price,
            median_price=TRAIN_STATS['median_price'],
            price_std=TRAIN_STATS['price_std']
        )
        interpretation = call_llm(interpretation_prompt, temperature=0.3)   # Slightly creative

        if not interpretation:
            interpretation = "No interpretation could be generated."

    except Exception as e:
        print("=== LLM INTERPRETATION ERROR ===")
        print(str(e))
        # Interpretation is non‑critical – fallback to a generic message
        interpretation = "Interpretation service is temporarily unavailable."

    # ------------------------------------------------------------------
    return PredictionResponse(
        extracted=extracted,
        predicted_price=pred_price,
        interpretation=interpretation
    )

# ------------------------------------------------------------------
# Health check
# ------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}
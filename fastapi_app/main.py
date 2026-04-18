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


# Load .env from project root (absolute path)
env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    # Use utf-8-sig to handle any BOM character
    with open(env_path, encoding='utf-8-sig') as f:
        load_dotenv(stream=f)
    print(".env file loaded successfully.")
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

    # Stage 1: Feature Extraction
    try:
        extraction_prompt = EXTRACTION_PROMPT.format(query=query)
        content = call_llm(extraction_prompt, temperature=0.0)

        if not content:
            raise ValueError("Groq returned an empty response")

        # Extract JSON object
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            json_str = content.strip()

        extracted_dict = json.loads(json_str)

        # Apply manual overrides if provided
        if request.manual_overrides:
            for key, value in request.manual_overrides.items():
                if key in extracted_dict:
                    extracted_dict[key] = value
                    if 'completeness' in extracted_dict and key in extracted_dict['completeness']:
                        extracted_dict['completeness'][key] = True
                    if 'missing_features' in extracted_dict and key in extracted_dict['missing_features']:
                        extracted_dict['missing_features'].remove(key)

        extracted = ExtractedFeatures(**extracted_dict)

    except Exception as e:
        print("=== LLM EXTRACTION ERROR ===")
        print(str(e))
        raise HTTPException(status_code=422, detail=f"LLM extraction failed: {str(e)}")

    # Stage 2: ML Prediction
    input_data = extracted.dict(by_alias=True, exclude={'completeness', 'missing_features'})
    input_df = pd.DataFrame([input_data])
    input_df = input_df.replace({None: np.nan})
    expected_cols = ml_model.feature_names_in_
    input_df = input_df[expected_cols]

    pred_log = ml_model.predict(input_df)[0]
    pred_price = float(np.expm1(pred_log))

    # Stage 3: Interpretation
    try:
        interp_prompt = INTERPRETATION_PROMPT.format(
            features=json.dumps(input_data, indent=2),
            prediction=pred_price,
            median_price=TRAIN_STATS['median_price'],
            price_std=TRAIN_STATS['price_std']
        )
        interpretation = call_llm(interp_prompt, temperature=0.3)
    except Exception as e:
        print("=== LLM INTERPRETATION ERROR (using fallback) ===")
        print(str(e))
        median = TRAIN_STATS['median_price']
        std = TRAIN_STATS['price_std']
        if pred_price > median + std:
            interpretation = f"The predicted price of ${pred_price:,.2f} is significantly above the market median (${median:,.2f}). This suggests the property has desirable features like larger living area or better quality."
        elif pred_price < median - std:
            interpretation = f"The predicted price of ${pred_price:,.2f} is below the market median (${median:,.2f}). This may reflect smaller size or lower condition ratings."
        else:
            interpretation = f"The predicted price of ${pred_price:,.2f} is within the typical range (median ${median:,.2f}). The property appears to be a standard offering for the area."

    # THIS WAS MISSING!
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
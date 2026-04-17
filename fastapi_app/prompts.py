# Two prompt variants for extraction (requirement 08)
EXTRACTION_PROMPT_V1 = """
Extract real estate features from the following description. Return a JSON object with these keys exactly:
- "Gr Liv Area" (number, sqft)
- "Garage Area" (number, sqft)
- "Year Built" (integer)
- "Total Bsmt SF" (number, sqft)
- "Lot Area" (number, sqft)
- "Overall Qual" (integer 1-10)
- "Overall Cond" (integer 1-10)
- "Bsmt Qual" (string: None, Po, Fa, TA, Gd, Ex)
- "Neighborhood" (string)
- "MS Zoning" (string)
- "Sale Condition" (string)

If a feature is not mentioned, set its value to null. Also include:
- "completeness": an object with the same keys and boolean true if extracted, false if missing.
- "missing_features": a list of strings containing the names of keys that are null.

User description: {query}

Respond ONLY with valid JSON. No extra text.
"""

EXTRACTION_PROMPT_V2 = """
You are a precise property data extractor. From the user's description, identify values for these 11 features. Use null when uncertain.

Features:
- Gr Liv Area (sqft)
- Garage Area (sqft)
- Year Built (integer)
- Total Bsmt SF (sqft)
- Lot Area (sqft)
- Overall Qual (1-10)
- Overall Cond (1-10)
- Bsmt Qual (None, Po, Fa, TA, Gd, Ex)
- Neighborhood (string)
- MS Zoning (string)
- Sale Condition (string)

Return a JSON object with the exact keys as above (including spaces) plus:
- "completeness": true/false for each key
- "missing_features": list of null keys

Description: {query}
"""

# We'll use V1 as default; you can swap or make it configurable.
EXTRACTION_PROMPT = EXTRACTION_PROMPT_V1

INTERPRETATION_PROMPT = """
You are a real estate analyst. Given these property features:
{features}
The predicted sale price is ${prediction:,.2f}.
Market median price is ${median_price:,.2f} with standard deviation ${price_std:,.2f}.
Provide a concise interpretation (2-4 sentences) explaining whether the price is above/below median and what features likely influenced it.
"""
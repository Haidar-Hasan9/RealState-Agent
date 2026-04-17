EXTRACTION_PROMPT_V1 = """
You are an expert real estate data extractor. From the user's property description, extract the following 11 features. For features not explicitly stated, infer reasonable values based on typical properties matching the description (e.g., "3-bedroom ranch" → ~1500-1800 sqft living area, "big garage" → ~600-800 sqft, "good neighborhood" → "NAmes"). If you cannot infer a value, set it to null.

Return a JSON object with these exact keys (including spaces):
- "Gr Liv Area" (above grade living area in sqft, number)
- "Garage Area" (garage area in sqft, number)
- "Year Built" (year built, integer)
- "Total Bsmt SF" (total basement area in sqft, number)
- "Lot Area" (lot size in sqft, number)
- "Overall Qual" (overall quality 1-10, integer)
- "Overall Cond" (overall condition 1-10, integer)
- "Bsmt Qual" (basement quality: None, Po, Fa, TA, Gd, Ex)
- "Neighborhood" (name of neighborhood, string)
- "MS Zoning" (zoning classification, string)
- "Sale Condition" (sale condition, string)

Also include:
- "completeness": an object with the same keys, boolean true if you could extract or infer a value, false if you had to leave it null.
- "missing_features": a list of strings containing the keys that are null.

User description: {query}

Respond ONLY with valid JSON. No extra text.
"""
EXTRACTION_PROMPT_V2 = EXTRACTION_PROMPT_V1  # Use the same for now, or create a variant

EXTRACTION_PROMPT = EXTRACTION_PROMPT_V1
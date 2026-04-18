import os
import sys
import json
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from groq import Groq
from fastapi_app.prompts import EXTRACTION_PROMPT_V1, EXTRACTION_PROMPT_V2

api_key = os.getenv("GROQ_API_KEY", "your-key-here-if-needed-for-local")

client = Groq(api_key=api_key)

queries = [
    "3-bedroom ranch with a big garage in a good neighborhood",
    "Luxury condo with 2000 sqft, built in 2015, near downtown",
    "Fixer-upper with large lot, 2 beds, needs work"
]

def extract_json(text):
    """Extract the first valid JSON object from text."""

    if text.startswith("```"):
        lines = text.split('\n')

        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = '\n'.join(lines)
    
    # Try to find a JSON object
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return match.group(0)
    return text.strip()

def evaluate(prompt_template, version_name):
    scores = []
    for i, q in enumerate(queries):
        prompt = prompt_template.format(query=q)
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        content = resp.choices[0].message.content
        print(f"\n--- {version_name} Query {i+1} RAW RESPONSE ---")
        print(repr(content[:200] + "..." if len(content) > 200 else content))

        if not content:
            print("WARNING: Empty response!")
            scores.append(0)
            continue

        # Clean and extract JSON
        clean_content = extract_json(content)
        try:
            data = json.loads(clean_content)
            filled = sum(1 for k, v in data.items() if k not in ('completeness','missing_features') and v is not None)
            print(f"Extracted {filled} features successfully.")
            scores.append(filled)
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            print(f"Cleaned content: {clean_content[:200]}")
            scores.append(0)
    return sum(scores) / len(scores)

print("Evaluating prompt V1...")
v1_avg = evaluate(EXTRACTION_PROMPT_V1, "V1")
print("\nEvaluating prompt V2...")
v2_avg = evaluate(EXTRACTION_PROMPT_V2, "V2")

print("\n=== RESULTS ===")
print(f"V1 average features extracted: {v1_avg:.1f} out of 11")
print(f"V2 average features extracted: {v2_avg:.1f} out of 11")
print("Winner: " + ("V1" if v1_avg >= v2_avg else "V2"))
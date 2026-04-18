import streamlit as st
import requests
import json
import os

API_URL = os.getenv("API_URL", "realstate-agent-production.up.railway.app/predict")

st.set_page_config(page_title="AI Real Estate Agent", page_icon="🏡")
st.title("🏡 AI Real Estate Agent")
st.write("Describe a property and get an AI-powered price estimate with explanation.")

# Initialize session state
if "extracted" not in st.session_state:
    st.session_state.extracted = None
if "prediction_data" not in st.session_state:
    st.session_state.prediction_data = None
if "last_query" not in st.session_state:
    st.session_state.last_query = ""

query = st.text_area("Property Description", placeholder="e.g., 3-bedroom ranch with big garage in a good neighborhood")

if st.button("Extract & Predict"):
    if not query.strip():
        st.warning("Please enter a property description.")
    else:
        with st.spinner("Analyzing..."):
            try:
                resp = requests.post(API_URL, json={"query": query})
                if resp.status_code == 200:
                    data = resp.json()
                    st.session_state.extracted = data['extracted']
                    st.session_state.prediction_data = data
                    st.session_state.last_query = query
                    st.rerun()
                else:
                    st.error(f"API Error: {resp.text}")
            except Exception as e:
                st.error(f"Connection error: {str(e)}")

# Display results if we have them
if st.session_state.prediction_data is not None:
    data = st.session_state.prediction_data
    extracted = st.session_state.extracted

    st.success(f"Predicted Price: **${data['predicted_price']:,.2f}**")
    
    st.subheader("Extracted Features")
    st.json(extracted)
    
    missing = extracted.get('missing_features', [])
    if missing:
        st.warning(f"Missing features: {', '.join(missing)}")
    
    st.subheader("Manually Provide Missing Features")
    missing_input = st.text_input(
        "Enter missing values as key:value pairs separated by commas",
        placeholder="e.g., Year Built:1995, Lot Area:8000, Bsmt Qual:TA"
    )
    
    if st.button("Update Prediction with Manual Values"):
        manual_updates = {}
        if missing_input.strip():
            try:
                for pair in missing_input.split(','):
                    if ':' not in pair:
                        continue
                    key, val = pair.split(':', 1)
                    key = key.strip()
                    val = val.strip()
                    # Convert to number if possible
                    if val.isdigit():
                        val = int(val)
                    else:
                        try:
                            val = float(val)
                        except:
                            pass
                    manual_updates[key] = val
            except Exception as e:
                st.error(f"Error parsing input: {e}")
                st.stop()
        
        payload = {
            "query": st.session_state.last_query,
            "manual_overrides": manual_updates
        }
        
        with st.spinner("Updating prediction..."):
            try:
                resp = requests.post(API_URL, json=payload)
                if resp.status_code == 200:
                    new_data = resp.json()
                    st.session_state.extracted = new_data['extracted']
                    st.session_state.prediction_data = new_data
                    st.rerun()
                else:
                    st.error(f"API Error: {resp.text}")
            except Exception as e:
                st.error(f"Connection error: {str(e)}")
    
    st.subheader("Interpretation")
    st.write(data['interpretation'])
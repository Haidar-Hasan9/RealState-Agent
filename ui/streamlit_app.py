import streamlit as st
import requests
import json
import os

API_URL = os.getenv("API_URL", "http://localhost:8000/predict")

st.set_page_config(page_title="AI Real Estate Agent", page_icon="🏡")
st.title("🏡 AI Real Estate Agent")
st.write("Describe a property and get an AI-powered price estimate with explanation.")

query = st.text_area("Property Description", placeholder="e.g., 3-bedroom ranch with big garage in a good neighborhood")

if st.button("Predict Price"):
    if not query.strip():
        st.warning("Please enter a property description.")
    else:
        with st.spinner("Analyzing..."):
            try:
                resp = requests.post(API_URL, json={"query": query})
                if resp.status_code == 200:
                    data = resp.json()
                    st.success(f"Predicted Price: **${data['predicted_price']:,.2f}**")
                    
                    # Show extracted features with completeness
                    st.subheader("Extracted Features")
                    extracted = data['extracted']
                    st.json(extracted)
                    
                    # Highlight missing features
                    missing = extracted.get('missing_features', [])
                    if missing:
                        st.warning(f"Missing features: {', '.join(missing)}")
                        st.info("You can manually provide these values in a real implementation.")
                    
                    st.subheader("Interpretation")
                    st.write(data['interpretation'])
                else:
                    st.error(f"API Error: {resp.text}")
            except Exception as e:
                st.error(f"Connection error: {str(e)}")
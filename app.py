# streamlit_app/app.py

import sys
from pathlib import Path

# Add the project root directory to Python path using pathlib
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import streamlit as st
import pandas as pd
from pipeline.scam_detector.detector import ScamDetector
from evaluator import evaluate_model

st.set_page_config(page_title="Scam Detection App", layout="wide")
st.title("Scam Detection")

# Detector instance
detector = ScamDetector()

# Tab layout
tab1, tab2 = st.tabs(["Single Message", "Dataset Evaluation"])

# ----------- Single Message Analysis ----------
with tab1:
    st.header("Analyze a Single Message")
    user_input = st.text_area("Enter the message to analyze:", height=150, 
                             placeholder="Example: Congratulations! You've won $1000. Click here to claim...")

    if st.button("Analyze Message", type="primary"):
        if user_input.strip() == "":
            st.warning("Please enter a message to analyze.")
        else:
            with st.spinner("Analyzing..."):
                result = detector.detect(user_input)
            
            st.success("Analysis completed!")
            
            # Display results
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Main prediction
                label = result.get("label", "Uncertain")
                if label == "Scam":
                    st.error(f"**PREDICTION: {label}**")
                elif label == "Not Scam":
                    st.success(f"**PREDICTION: {label}**")
                else:
                    st.warning(f"**PREDICTION: {label}**")
                
                # Intent
                intent = result.get("intent", "Unknown")
                st.info(f"**Intent Detected:** {intent}")
            
            with col2:
                # Risk factors
                risk_factors = result.get("risk_factors", [])
              

# ----------- Dataset Evaluation ----------
with tab2:
    st.header("Evaluate Model on Dataset")
    st.write("Upload a CSV file with columns: 'text' (or 'message_text') and 'label'")
    
    uploaded_file = st.file_uploader("Choose CSV file", type=["csv"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Check for required columns
            text_col = None
            if 'text' in df.columns:
                text_col = 'text'
            elif 'message_text' in df.columns:
                text_col = 'message_text'
            
            if text_col is None or 'label' not in df.columns:
                st.error("CSV must contain 'text' (or 'message_text') and 'label' columns")
            else:
                st.success(f"Dataset loaded: {len(df)} messages")
                
                # Show sample data
                with st.expander("Sample Data", expanded=False):
                    st.dataframe(df.head())
                
                # Evaluation controls
                col1, col2 = st.columns(2)
                with col1:
                    limit = st.number_input("Limit messages (for testing)", 
                                          min_value=1, max_value=len(df), 
                                          value=min(50, len(df)))
                with col2:
                    if st.button("Evaluate Dataset", type="primary"):
                        with st.spinner("Processing messages in batches..."):
                            try:
                                # Use batch processing instead of individual calls
                                messages = df[text_col].tolist()[:limit]
                                actual_labels = df['label'].tolist()[:limit]
                                
                                # Simple batch processing
                                predicted_results = detector.detect_batch(messages)
                                predicted_labels = [result['label'] for result in predicted_results]
                                
                                # Calculate results using our simple metrics
                                results = evaluate_model(actual_labels, predicted_labels)
                                
                                # Display results
                                st.success("Evaluation completed!")
                                
                                # Main metrics
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Overall Accuracy", f"{results['overall_accuracy']}%")
                                with col2:
                                    st.metric("Total Predictions", results['total_predictions'])
                                with col3:
                                    st.metric("Correct Predictions", results['correct_predictions'])
                                
                                # Summary
                                st.info(f"**Summary:** {results['summary']}")
                                
                            except Exception as e:
                                st.error(f"Error processing dataset: {str(e)}")
        except Exception as e:
            st.error(f"Error loading CSV file: {str(e)}")

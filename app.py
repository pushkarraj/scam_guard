# streamlit_app/app.py

import sys
from pathlib import Path
from collections import Counter

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pipeline.scam_detector.detector import ScamDetector
from evaluator import calculate_metrics

st.set_page_config(page_title="Scam Detection App", layout="wide")

st.markdown("""
    <style>
        .result-card {
            border-radius: 12px;
            padding: 24px 28px;
            margin-bottom: 16px;
        }
        .result-scam    { background: #ffe5e5; border-left: 6px solid #e53935; }
        .result-notscam { background: #e6f9ee; border-left: 6px solid #2e7d32; }
        .result-uncertain { background: #fff8e1; border-left: 6px solid #f9a825; }
        .result-label { font-size: 1.6rem; font-weight: 700; margin: 0; }
        .label-scam     { color: #c62828; }
        .label-notscam  { color: #1b5e20; }
        .label-uncertain { color: #e65100; }
        .pill {
            display: inline-block;
            background: #ff5252;
            color: white;
            border-radius: 20px;
            padding: 3px 12px;
            font-size: 0.78rem;
            font-weight: 600;
            margin: 3px 4px 3px 0;
        }
        .intent-chip {
            display: inline-block;
            background: #e3f2fd;
            color: #1565c0;
            border-radius: 20px;
            padding: 4px 14px;
            font-size: 0.85rem;
            font-weight: 600;
        }
        .section-title {
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #888;
            margin-bottom: 6px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Scam Detection")

detector = ScamDetector()

tab1, tab2 = st.tabs(["Single Message", "Dataset Evaluation"])


# ─── Tab 1: Single Message ──────────────────────────────────────────────────
with tab1:
    st.header("Analyze a Single Message")
    user_input = st.text_area(
        "Enter the message to analyze:",
        height=150,
        placeholder="Example: Congratulations! You've won $1000. Click here to claim...",
    )

    if st.button("Analyze Message", type="primary"):
        if user_input.strip() == "":
            st.warning("Please enter a message to analyze.")
        else:
            with st.spinner("Analyzing..."):
                result = detector.detect(user_input)

            label       = result.get("label", "Uncertain")
            intent      = result.get("intent", "Unknown")
            risk_factors = result.get("risk_factors", [])
            reasoning   = result.get("reasoning", "")

            # ── Result banner ──
            if label == "Scam":
                card_cls, label_cls, icon = "result-scam", "label-scam", "🚨"
            elif label == "Not Scam":
                card_cls, label_cls, icon = "result-notscam", "label-notscam", "✅"
            else:
                card_cls, label_cls, icon = "result-uncertain", "label-uncertain", "⚠️"

            st.markdown(
                f'<div class="result-card {card_cls}">'
                f'<p class="result-label {label_cls}">{icon} {label}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

            col1, col2 = st.columns(2)

            with col1:
                st.markdown('<p class="section-title">Detected Intent</p>', unsafe_allow_html=True)
                st.markdown(f'<span class="intent-chip">🎯 {intent}</span>', unsafe_allow_html=True)

            with col2:
                if risk_factors:
                    st.markdown('<p class="section-title">Risk Factors</p>', unsafe_allow_html=True)
                    pills_html = "".join(
                        f'<span class="pill">⚑ {rf}</span>' for rf in risk_factors
                    )
                    st.markdown(pills_html, unsafe_allow_html=True)

            if reasoning:
                with st.expander("View Reasoning"):
                    st.write(reasoning)


# ─── Tab 2: Dataset Evaluation ───────────────────────────────────────────────
with tab2:
    st.header("Evaluate Model on Dataset")
    st.write("Upload a CSV file with columns: `text` (or `message_text`) and `label`")

    uploaded_file = st.file_uploader("Choose CSV file", type=["csv"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)

            text_col = None
            if "text" in df.columns:
                text_col = "text"
            elif "message_text" in df.columns:
                text_col = "message_text"

            if text_col is None or "label" not in df.columns:
                st.error("CSV must contain 'text' (or 'message_text') and 'label' columns")
            else:
                st.success(f"Dataset loaded: {len(df)} messages")

                with st.expander("Sample Data", expanded=False):
                    st.dataframe(df.head())

                col1, col2 = st.columns(2)
                with col1:
                    limit = st.number_input(
                        "Limit messages (for testing)",
                        min_value=1, max_value=len(df),
                        value=min(50, len(df)),
                    )
                with col2:
                    run_eval = st.button("Evaluate Dataset", type="primary")

                if run_eval:
                    with st.spinner("Processing messages in batches..."):
                        try:
                            messages      = df[text_col].tolist()[:limit]
                            actual_labels = df["label"].tolist()[:limit]

                            predicted_results = detector.detect_batch(messages)
                            predicted_labels  = [r["label"] for r in predicted_results]

                            results = calculate_metrics(actual_labels, predicted_labels)

                            st.success("Evaluation completed!")
                            st.divider()

                            # ── Metric cards ──────────────────────────────
                            c1, c2, c3 = st.columns(3)
                            c1.metric("Overall Accuracy",    f"{results['accuracy']}%")
                            c2.metric("Total Predictions",   results["total"])
                            c3.metric("Correct Predictions", results["correct"])

                            st.divider()

                            # ── Accuracy gauge ────────────────────────────
                            acc = results["accuracy"]
                            gauge_color = (
                                "#2e7d32" if acc >= 80
                                else "#f9a825" if acc >= 60
                                else "#e53935"
                            )
                            gauge_fig = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=acc,
                                number={"suffix": "%", "font": {"size": 36}},
                                title={"text": "Overall Accuracy", "font": {"size": 16}},
                                gauge={
                                    "axis": {"range": [0, 100], "tickwidth": 1},
                                    "bar":  {"color": gauge_color, "thickness": 0.25},
                                    "bgcolor": "white",
                                    "steps": [
                                        {"range": [0, 60],  "color": "#ffebee"},
                                        {"range": [60, 80], "color": "#fff8e1"},
                                        {"range": [80, 100],"color": "#e8f5e9"},
                                    ],
                                    "threshold": {
                                        "line": {"color": gauge_color, "width": 4},
                                        "thickness": 0.75,
                                        "value": acc,
                                    },
                                },
                            ))
                            gauge_fig.update_layout(height=280, margin=dict(t=40, b=0, l=30, r=30))
                            st.plotly_chart(gauge_fig, use_container_width=True)

                            st.divider()

                            # ── Per-class bar chart + prediction pie ──────
                            chart_col1, chart_col2 = st.columns(2)

                            with chart_col1:
                                class_metrics = results.get("class_metrics", {})
                                if class_metrics:
                                    classes  = [c.title() for c in class_metrics]
                                    recalls  = [class_metrics[c]["recall"]   for c in class_metrics]
                                    f1_scores= [class_metrics[c]["f1_score"] for c in class_metrics]

                                    bar_fig = go.Figure(data=[
                                        go.Bar(
                                            name="Recall %",
                                            x=classes, y=recalls,
                                            marker_color="#1565c0",
                                            text=[f"{v:.1f}%" for v in recalls],
                                            textposition="outside",
                                        ),
                                        go.Bar(
                                            name="F1-Score %",
                                            x=classes, y=f1_scores,
                                            marker_color="#2e7d32",
                                            text=[f"{v:.1f}%" for v in f1_scores],
                                            textposition="outside",
                                        ),
                                    ])
                                    bar_fig.update_layout(
                                        title="Per-Class Recall & F1-Score",
                                        barmode="group",
                                        yaxis=dict(range=[0, 110], title="Score (%)"),
                                        legend=dict(orientation="h", y=-0.2),
                                        height=340,
                                        margin=dict(t=50, b=20, l=20, r=20),
                                    )
                                    st.plotly_chart(bar_fig, use_container_width=True)

                            with chart_col2:
                                label_counts = Counter(predicted_labels)
                                color_map = {
                                    "Scam":     "#e53935",
                                    "Not Scam": "#2e7d32",
                                    "Uncertain":"#f9a825",
                                }
                                pie_labels = list(label_counts.keys())
                                pie_values = list(label_counts.values())
                                pie_colors = [color_map.get(l, "#90a4ae") for l in pie_labels]

                                pie_fig = go.Figure(go.Pie(
                                    labels=pie_labels,
                                    values=pie_values,
                                    hole=0.45,
                                    marker_colors=pie_colors,
                                    textinfo="label+percent",
                                    hovertemplate="%{label}: %{value} messages<extra></extra>",
                                ))
                                pie_fig.update_layout(
                                    title="Prediction Distribution",
                                    height=340,
                                    margin=dict(t=50, b=20, l=20, r=20),
                                    legend=dict(orientation="h", y=-0.1),
                                )
                                st.plotly_chart(pie_fig, use_container_width=True)

                        except Exception as e:
                            st.error(f"Error processing dataset: {str(e)}")
        except Exception as e:
            st.error(f"Error loading CSV file: {str(e)}")

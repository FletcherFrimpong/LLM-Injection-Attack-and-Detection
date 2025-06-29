"""
SecPrompt Dashboard

A Streamlit web application for interactive prompt injection analysis.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import sys
from pathlib import Path

# Add the secprompt package to the path
sys.path.append(str(Path(__file__).parent.parent))

from secprompt.detector import PromptDetector
from secprompt.evaluator import PromptEvaluator
from secprompt.defenses import PromptDefender
from secprompt.simulator import PromptSimulator


def main():
    st.set_page_config(
        page_title="SecPrompt Dashboard",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🛡️ SecPrompt - Prompt Injection Security Dashboard")
    st.markdown("Comprehensive analysis and defense against prompt injection attacks")
    
    # Initialize components
    if 'detector' not in st.session_state:
        st.session_state.detector = PromptDetector()
    if 'evaluator' not in st.session_state:
        st.session_state.evaluator = PromptEvaluator()
    if 'defender' not in st.session_state:
        st.session_state.defender = PromptDefender()
    if 'simulator' not in st.session_state:
        st.session_state.simulator = PromptSimulator()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Single Analysis", "Batch Analysis", "Defense Testing", "Model Training", "Data Generation"]
    )
    
    if page == "Single Analysis":
        single_analysis_page()
    elif page == "Batch Analysis":
        batch_analysis_page()
    elif page == "Defense Testing":
        defense_testing_page()
    elif page == "Model Training":
        model_training_page()
    elif page == "Data Generation":
        data_generation_page()


def single_analysis_page():
    st.header("Single Prompt Analysis")
    
    # Input section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        text_input = st.text_area(
            "Enter text to analyze",
            placeholder="Enter your text here...",
            height=150
        )
    
    with col2:
        st.subheader("Analysis Options")
        use_ml = st.checkbox("Use ML Model", value=False)
        production_context = st.checkbox("Production Environment", value=False)
        sensitive_data = st.checkbox("Sensitive Data Context", value=False)
    
    if st.button("Analyze", type="primary"):
        if text_input.strip():
            analyze_single_prompt(text_input, use_ml, production_context, sensitive_data)
        else:
            st.warning("Please enter some text to analyze.")


def analyze_single_prompt(text, use_ml, production_context, sensitive_data):
    st.subheader("Analysis Results")
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs(["Detection", "Evaluation", "Defense", "Raw Data"])
    
    with tab1:
        st.subheader("🔍 Detection Results")
        
        if use_ml and st.session_state.detector.is_trained:
            result = st.session_state.detector.predict(text)
        else:
            result = st.session_state.detector.rule_based_detection(text)
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if result.is_injection:
                st.error("🚨 INJECTION DETECTED")
            else:
                st.success("✅ SAFE")
        
        with col2:
            st.metric("Confidence", f"{result.confidence:.2%}")
        
        with col3:
            st.metric("Category", result.category.replace("_", " ").title())
        
        st.write("**Explanation:**", result.explanation)
        
        # Feature importance
        if hasattr(result, 'features') and result.features:
            st.subheader("Feature Analysis")
            features_df = pd.DataFrame(list(result.features.items()), columns=['Feature', 'Value'])
            st.bar_chart(features_df.set_index('Feature'))
    
    with tab2:
        st.subheader("📊 Evaluation Results")
        
        context = {}
        if production_context:
            context["production_environment"] = True
        if sensitive_data:
            context["sensitive_data"] = True
        
        eval_result = st.session_state.evaluator.evaluate_prompt(text, context)
        
        # Severity indicator
        severity_colors = {
            "low": "green",
            "medium": "orange", 
            "high": "red",
            "critical": "darkred"
        }
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Severity", eval_result.severity.value.upper())
        
        with col2:
            st.metric("Impact Score", f"{eval_result.impact_score:.2%}")
        
        with col3:
            st.metric("Confidence", f"{eval_result.confidence:.2%}")
        
        # Impact types
        if eval_result.impact_types:
            st.subheader("Impact Types")
            impact_types = [t.value.replace("_", " ").title() for t in eval_result.impact_types]
            for impact in impact_types:
                st.write(f"• {impact}")
        
        # Risk factors
        if eval_result.risk_factors:
            st.subheader("Risk Factors")
            for factor in eval_result.risk_factors:
                st.write(f"• {factor}")
        
        # Recommendations
        if eval_result.recommendations:
            st.subheader("Recommendations")
            for rec in eval_result.recommendations:
                st.write(f"• {rec}")
    
    with tab3:
        st.subheader("🛡️ Defense Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Input Validation")
            validation = st.session_state.defender.validate_input(text)
            
            if validation["is_safe"]:
                st.success("✅ Input appears safe")
            else:
                st.error("🚨 Suspicious input detected")
            
            st.metric("Risk Score", f"{validation['risk_score']:.2%}")
            
            if validation["detected_patterns"]:
                st.write("**Detected Patterns:**")
                for pattern in validation["detected_patterns"]:
                    st.write(f"• {pattern['category']}: {pattern['severity']}")
        
        with col2:
            st.subheader("Sanitization Preview")
            sanitize_aggressive = st.checkbox("Aggressive Sanitization", value=False)
            
            if st.button("Apply Sanitization"):
                sanitize_result = st.session_state.defender.sanitize_input(text, sanitize_aggressive)
                
                st.write("**Original:**")
                st.code(text)
                st.write("**Sanitized:**")
                st.code(sanitize_result.sanitized_text)
                
                if sanitize_result.removed_content:
                    st.write("**Removed Content:**")
                    for item in sanitize_result.removed_content:
                        st.write(f"• {item}")
    
    with tab4:
        st.subheader("📋 Raw Analysis Data")
        
        # Detection data
        st.write("**Detection Data:**")
        detection_data = {
            "is_injection": result.is_injection,
            "confidence": result.confidence,
            "category": result.category,
            "severity": result.severity,
            "explanation": result.explanation
        }
        st.json(detection_data)
        
        # Evaluation data
        st.write("**Evaluation Data:**")
        eval_data = {
            "severity": eval_result.severity.value,
            "impact_score": eval_result.impact_score,
            "impact_types": [t.value for t in eval_result.impact_types],
            "risk_factors": eval_result.risk_factors,
            "confidence": eval_result.confidence,
            "recommendations": eval_result.recommendations
        }
        st.json(eval_data)


def batch_analysis_page():
    st.header("Batch Analysis")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload a JSON file with texts to analyze",
        type=['json', 'txt'],
        help="Upload a JSON file with 'text' field or a text file with one text per line"
    )
    
    # Or use sample data
    use_sample = st.checkbox("Use sample data", value=True)
    
    if st.button("Run Batch Analysis", type="primary"):
        if uploaded_file or use_sample:
            run_batch_analysis(uploaded_file, use_sample)
        else:
            st.warning("Please upload a file or use sample data.")


def run_batch_analysis(uploaded_file, use_sample):
    # Load data
    if use_sample:
        sample_file = Path(__file__).parent.parent / "data" / "sample_payloads.json"
        with open(sample_file, 'r') as f:
            data = json.load(f)
        texts = [item['content'] for item in data]
        labels = [1 if item['category'] != 'benign' else 0 for item in data]
    elif uploaded_file:
        if uploaded_file.name.endswith('.json'):
            data = json.load(uploaded_file)
            texts = [item['text'] for item in data]
            labels = [item.get('is_injection', 0) for item in data]
        else:
            texts = [line.strip() for line in uploaded_file if line.strip()]
            labels = [0] * len(texts)  # Assume all safe
    
    st.subheader(f"Analyzing {len(texts)} texts...")
    
    # Run analysis
    detection_results = []
    evaluation_results = []
    
    progress_bar = st.progress(0)
    
    for i, text in enumerate(texts):
        # Detection
        det_result = st.session_state.detector.rule_based_detection(text)
        detection_results.append(det_result)
        
        # Evaluation
        eval_result = st.session_state.evaluator.evaluate_prompt(text)
        evaluation_results.append(eval_result)
        
        progress_bar.progress((i + 1) / len(texts))
    
    # Display results
    display_batch_results(texts, detection_results, evaluation_results, labels)


def display_batch_results(texts, detection_results, evaluation_results, labels):
    # Create summary dataframe
    data = []
    for i, (text, det, eval_res) in enumerate(zip(texts, detection_results, evaluation_results)):
        data.append({
            'Text': text[:50] + '...' if len(text) > 50 else text,
            'Detection': 'INJECTION' if det.is_injection else 'SAFE',
            'Confidence': det.confidence,
            'Category': det.category,
            'Severity': eval_res.severity.value,
            'Impact Score': eval_res.impact_score,
            'Actual': 'INJECTION' if labels[i] else 'SAFE' if i < len(labels) else 'UNKNOWN'
        })
    
    df = pd.DataFrame(data)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_injections = sum(1 for r in detection_results if r.is_injection)
        st.metric("Detected Injections", total_injections)
    
    with col2:
        avg_confidence = sum(r.confidence for r in detection_results) / len(detection_results)
        st.metric("Avg Confidence", f"{avg_confidence:.2%}")
    
    with col3:
        avg_impact = sum(r.impact_score for r in evaluation_results) / len(evaluation_results)
        st.metric("Avg Impact Score", f"{avg_impact:.2%}")
    
    with col4:
        critical_count = sum(1 for r in evaluation_results if r.severity.value == 'critical')
        st.metric("Critical Severity", critical_count)
    
    # Results table
    st.subheader("Detailed Results")
    st.dataframe(df, use_container_width=True)
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Severity distribution
        severity_counts = df['Severity'].value_counts()
        fig = px.pie(values=severity_counts.values, names=severity_counts.index, title="Severity Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Category distribution
        category_counts = df['Category'].value_counts()
        fig = px.bar(x=category_counts.index, y=category_counts.values, title="Category Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    # Impact score distribution
    fig = px.histogram(df, x='Impact Score', title="Impact Score Distribution")
    st.plotly_chart(fig, use_container_width=True)


def defense_testing_page():
    st.header("Defense Testing")
    
    # Input
    text = st.text_area("Enter text to test defenses", placeholder="Enter malicious text...")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Defense Options")
        mode = st.selectbox("Defense Mode", ["sanitize", "rewrite"])
        aggressive = st.checkbox("Aggressive Sanitization", value=False)
        
        # Rewrite options
        if mode == "rewrite":
            add_isolation = st.checkbox("Add Context Isolation", value=True)
            add_reinforcement = st.checkbox("Add Instruction Reinforcement", value=True)
            add_validation = st.checkbox("Add Validation Instructions", value=True)
            add_monitoring = st.checkbox("Add Monitoring Instructions", value=True)
    
    with col2:
        st.subheader("Validation Results")
        if text:
            validation = st.session_state.defender.validate_input(text)
            
            if validation["is_safe"]:
                st.success("✅ Input appears safe")
            else:
                st.error("🚨 Suspicious input detected")
            
            st.metric("Risk Score", f"{validation['risk_score']:.2%}")
            
            if validation["warnings"]:
                st.write("**Warnings:**")
                for warning in validation["warnings"]:
                    st.write(f"• {warning}")
    
    if st.button("Apply Defenses", type="primary") and text:
        if mode == "sanitize":
            result = st.session_state.defender.sanitize_input(text, aggressive)
        else:
            context = {
                "add_isolation": add_isolation,
                "add_reinforcement": add_reinforcement,
                "add_validation": add_validation,
                "add_monitoring": add_monitoring
            }
            result = st.session_state.defender.rewrite_prompt(text, context)
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Text")
            st.code(text)
        
        with col2:
            st.subheader("Defended Text")
            st.code(result.sanitized_text)
        
        st.subheader("Defense Details")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Confidence", f"{result.confidence:.2%}")
        
        with col2:
            st.metric("Applied Defenses", len(result.applied_defenses))
        
        with col3:
            st.metric("Removed Items", len(result.removed_content))
        
        if result.applied_defenses:
            st.write("**Applied Defenses:**")
            for defense in result.applied_defenses:
                st.write(f"• {defense.value.replace('_', ' ').title()}")
        
        if result.removed_content:
            st.write("**Removed Content:**")
            for item in result.removed_content:
                st.write(f"• {item}")
        
        if result.warnings:
            st.write("**Warnings:**")
            for warning in result.warnings:
                st.write(f"• {warning}")


def model_training_page():
    st.header("Model Training")
    
    st.info("Model training functionality will be implemented in a future version.")
    st.write("This page will allow you to:")
    st.write("• Upload training data")
    st.write("• Select model type")
    st.write("• Train and evaluate models")
    st.write("• Save trained models")


def data_generation_page():
    st.header("Data Generation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Generation Options")
        size = st.slider("Number of payloads", 10, 500, 100)
        include_mutations = st.checkbox("Include mutations", value=True)
        category_filter = st.multiselect(
            "Categories to include",
            ["role_confusion", "instruction_override", "context_manipulation", "encoding_evasion"],
            default=["role_confusion", "instruction_override", "context_manipulation", "encoding_evasion"]
        )
    
    with col2:
        st.subheader("Preview")
        if st.button("Generate Sample", type="primary"):
            dataset = st.session_state.simulator.generate_dataset(size=10, include_mutations=False)
            
            preview_data = []
            for payload in dataset:
                preview_data.append({
                    'Content': payload.content[:50] + '...' if len(payload.content) > 50 else payload.content,
                    'Category': payload.category,
                    'Severity': payload.severity
                })
            
            df = pd.DataFrame(preview_data)
            st.dataframe(df, use_container_width=True)
    
    if st.button("Generate Full Dataset", type="primary"):
        with st.spinner("Generating dataset..."):
            dataset = st.session_state.simulator.generate_dataset(size=size, include_mutations=include_mutations)
        
        # Save dataset
        output_file = f"data/generated_payloads_{size}.json"
        st.session_state.simulator.save_dataset(dataset, output_file)
        
        st.success(f"Generated {len(dataset)} payloads and saved to {output_file}")
        
        # Show statistics
        categories = [p.category for p in dataset]
        severities = [p.severity for p in dataset]
        
        col1, col2 = st.columns(2)
        
        with col1:
            category_counts = pd.Series(categories).value_counts()
            fig = px.pie(values=category_counts.values, names=category_counts.index, title="Category Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            severity_counts = pd.Series(severities).value_counts()
            fig = px.bar(x=severity_counts.index, y=severity_counts.values, title="Severity Distribution")
            st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main() 
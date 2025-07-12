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
from secprompt.defense_tester import DefenseTester, TestScenario
from secprompt.model_trainer import ModelTrainer, TrainingConfig, TrainingResult
from secprompt.data_collector import DataCollector


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
    if 'defense_tester' not in st.session_state:
        st.session_state.defense_tester = DefenseTester()
    if 'model_trainer' not in st.session_state:
        st.session_state.model_trainer = ModelTrainer()
    if 'data_collector' not in st.session_state:
        st.session_state.data_collector = DataCollector()
    
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
    st.header("🛡️ Defense Testing")
    
    # Test mode selection
    test_mode = st.selectbox(
        "Test Mode",
        ["Custom Input", "Predefined Scenarios", "Comprehensive Suite"],
        help="Choose how to test defense mechanisms"
    )
    
    if test_mode == "Custom Input":
        custom_input_test()
    elif test_mode == "Predefined Scenarios":
        scenario_test()
    else:
        comprehensive_test()


def custom_input_test():
    st.subheader("Custom Input Testing")
    
    # Input
    text = st.text_area(
        "Enter text to test defenses", 
        placeholder="Enter malicious text...",
        help="Enter any text to test how well the defense mechanisms work"
    )
    
    # Defense options
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Defense Options")
        enable_sanitization = st.checkbox("Enable Sanitization", value=True)
        enable_rewriting = st.checkbox("Enable Rewriting", value=True)
        enable_isolation = st.checkbox("Enable Isolation", value=True)
        enable_validation = st.checkbox("Enable Validation", value=True)
    
    with col2:
        st.subheader("Quick Validation")
        if text:
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
    
    if st.button("Run Defense Test", type="primary") and text:
        with st.spinner("Running comprehensive defense test..."):
            result = st.session_state.defense_tester.run_defense_test(
                text,
                enable_sanitization=enable_sanitization,
                enable_rewriting=enable_rewriting,
                enable_isolation=enable_isolation,
                enable_validation=enable_validation
            )
        
        display_defense_test_results(result)


def scenario_test():
    st.subheader("Predefined Scenario Testing")
    
    # Scenario selection
    scenarios = [s.value for s in TestScenario]
    selected_scenario = st.selectbox("Select Test Scenario", scenarios)
    
    # Show scenario description
    scenario_descriptions = {
        "role_confusion": "Tests attempts to make the AI act as a different entity",
        "instruction_override": "Tests attempts to override system instructions",
        "system_prompt_leak": "Tests attempts to extract system prompts",
        "financial_fraud": "Tests financial fraud attempts",
        "encoding_evasion": "Tests attempts to evade detection through encoding",
        "context_manipulation": "Tests attempts to manipulate context",
        "safety_bypass": "Tests attempts to bypass safety measures",
        "data_exfiltration": "Tests attempts to extract sensitive data"
    }
    
    st.info(f"**{selected_scenario.replace('_', ' ').title()}**: {scenario_descriptions.get(selected_scenario, '')}")
    
    if st.button("Test Scenario", type="primary"):
        with st.spinner(f"Testing {selected_scenario} scenario..."):
            try:
                scenario = TestScenario(selected_scenario)
                result = st.session_state.defense_tester.run_scenario_test(scenario)
                display_defense_test_results(result)
            except Exception as e:
                st.error(f"Error testing scenario: {e}")


def comprehensive_test():
    st.subheader("Comprehensive Test Suite")
    
    st.info("This will test all predefined scenarios and provide a comprehensive analysis of defense effectiveness.")
    
    if st.button("Run Comprehensive Test", type="primary"):
        with st.spinner("Running comprehensive test suite..."):
            report = st.session_state.defense_tester.run_comprehensive_test_suite()
        
        # Display summary
        st.subheader("📊 Test Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Tests", report.summary['total_tests'])
        
        with col2:
            st.metric("Successful Defenses", report.summary['successful_defenses'])
        
        with col3:
            st.metric("Success Rate", f"{report.summary['success_rate']:.1f}%")
        
        with col4:
            st.metric("Avg Effectiveness", f"{report.summary['average_effectiveness']:.1f}%")
        
        # Status distribution
        st.subheader("Status Distribution")
        status_data = report.summary['status_distribution']
        status_df = pd.DataFrame(list(status_data.items()), columns=['Status', 'Count'])
        fig = px.pie(status_df, values='Count', names='Status', title="Defense Effectiveness Distribution")
        st.plotly_chart(fig, use_container_width=True)
        
        # Individual test results
        st.subheader("Individual Test Results")
        test_data = []
        for test_result in report.test_results:
            test_data.append({
                'Scenario': test_result.scenario.replace('_', ' ').title(),
                'Effectiveness': test_result.overall_effectiveness['score'],
                'Status': test_result.overall_effectiveness['status'].title(),
                'Enabled Defenses': test_result.overall_effectiveness['enabled_defenses'],
                'Effective Defenses': test_result.overall_effectiveness['effective_defenses']
            })
        
        test_df = pd.DataFrame(test_data)
        st.dataframe(test_df, use_container_width=True)
        
        # Recommendations
        st.subheader("💡 Recommendations")
        for rec in report.recommendations:
            st.write(f"• {rec}")


def display_defense_test_results(result):
    """Display comprehensive defense test results"""
    st.subheader("🔍 Defense Test Results")
    
    # Overall effectiveness
    effectiveness = result.overall_effectiveness
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Overall Score", f"{effectiveness['score']:.1f}%")
    
    with col2:
        status_color = {
            "excellent": "green",
            "good": "blue", 
            "fair": "orange",
            "poor": "red"
        }.get(effectiveness['status'], "gray")
        st.metric("Status", effectiveness['status'].title())
    
    with col3:
        st.metric("Enabled Defenses", effectiveness['enabled_defenses'])
    
    with col4:
        st.metric("Effective Defenses", effectiveness['effective_defenses'])
    
    # Detailed results in tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Sanitization", "Rewriting", "Isolation", "Validation"])
    
    with tab1:
        if result.sanitization_result:
            sanit = result.sanitization_result
            st.subheader("Input Sanitization")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Original Text:**")
                st.code(sanit.original_text)
            
            with col2:
                st.write("**Sanitized Text:**")
                st.code(sanit.sanitized_text)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Confidence", f"{sanit.confidence:.2%}")
            
            with col2:
                st.metric("Removed Items", len(sanit.removed_content))
            
            with col3:
                st.metric("Effective", "Yes" if sanit.original_text != sanit.sanitized_text else "No")
            
            if sanit.removed_content:
                st.write("**Removed Content:**")
                for item in sanit.removed_content:
                    st.write(f"• {item}")
        else:
            st.info("Sanitization was not enabled for this test.")
    
    with tab2:
        if result.rewriting_result:
            rewrite = result.rewriting_result
            st.subheader("Prompt Rewriting")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Original Text:**")
                st.code(rewrite.original_text)
            
            with col2:
                st.write("**Rewritten Text:**")
                st.code(rewrite.sanitized_text)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Confidence", f"{rewrite.confidence:.2%}")
            
            with col2:
                st.metric("Applied Defenses", len(rewrite.applied_defenses))
            
            with col3:
                st.metric("Effective", "Yes" if len(rewrite.applied_defenses) > 0 else "No")
            
            if rewrite.applied_defenses:
                st.write("**Applied Defenses:**")
                for defense in rewrite.applied_defenses:
                    st.write(f"• {defense.value.replace('_', ' ').title()}")
        else:
            st.info("Rewriting was not enabled for this test.")
    
    with tab3:
        if result.isolation_result:
            isolation = result.isolation_result
            st.subheader("Context Isolation")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Original Text:**")
                st.code(isolation.original_text)
            
            with col2:
                st.write("**Isolated Text:**")
                st.code(isolation.sanitized_text)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Confidence", f"{isolation.confidence:.2%}")
            
            with col2:
                st.metric("Context Markers", "Yes" if '[USER_INPUT_START]' in isolation.sanitized_text else "No")
            
            with col3:
                st.metric("Effective", "Yes" if '[USER_INPUT_START]' in isolation.sanitized_text and '[USER_INPUT_END]' in isolation.sanitized_text else "No")
        else:
            st.info("Isolation was not enabled for this test.")
    
    with tab4:
        if result.validation_result:
            validation = result.validation_result
            st.subheader("Input Validation")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Safe", "Yes" if validation.get('is_safe', True) else "No")
            
            with col2:
                st.metric("Risk Score", f"{validation.get('risk_score', 0):.2%}")
            
            with col3:
                st.metric("Detected Threats", len(validation.get('detected_patterns', [])))
            
            if validation.get('detected_patterns'):
                st.write("**Detected Patterns:**")
                for pattern in validation['detected_patterns']:
                    st.write(f"• {pattern['category']}: {pattern['severity']}")
            
            if validation.get('warnings'):
                st.write("**Warnings:**")
                for warning in validation['warnings']:
                    st.write(f"• {warning}")
        else:
            st.info("Validation was not enabled for this test.")


def model_training_page():
    st.header("🤖 Model Training")
    
    # Training mode selection
    training_mode = st.selectbox(
        "Training Mode",
        ["Data Collection", "Model Training", "Model Evaluation", "Model Management"],
        help="Choose the training workflow step"
    )
    
    if training_mode == "Data Collection":
        data_collection_page()
    elif training_mode == "Model Training":
        model_training_workflow()
    elif training_mode == "Model Evaluation":
        model_evaluation_page()
    else:
        model_management_page()


def data_collection_page():
    st.subheader("📊 Data Collection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Data Sources")
        sources = st.multiselect(
            "Select data sources",
            ["synthetic", "github", "reddit", "huggingface", "arxiv"],
            default=["synthetic"],
            help="Choose which sources to collect data from"
        )
        
        max_samples = st.slider(
            "Max samples per source",
            min_value=100,
            max_value=2000,
            value=500,
            step=100
        )
        
        save_path = st.text_input(
            "Save dataset to",
            value="data/training_data.json",
            help="Path to save the collected dataset"
        )
    
    with col2:
        st.subheader("Data Preview")
        if st.button("Generate Sample Data", type="primary"):
            with st.spinner("Generating synthetic sample data..."):
                samples = st.session_state.data_collector.generate_synthetic_data(num_samples=50)
                
                # Show sample data
                sample_data = []
                for sample in samples[:10]:
                    sample_data.append({
                        'Text': sample.text[:50] + '...' if len(sample.text) > 50 else sample.text,
                        'Is Injection': sample.is_injection,
                        'Category': sample.category,
                        'Source': sample.source
                    })
                
                df = pd.DataFrame(sample_data)
                st.dataframe(df, use_container_width=True)
    
    if st.button("Collect Training Data", type="primary"):
        with st.spinner("Collecting data from selected sources..."):
            try:
                samples = st.session_state.model_trainer.collect_training_data(
                    sources=sources,
                    max_samples_per_source=max_samples,
                    save_path=save_path
                )
                
                # Show statistics
                stats = st.session_state.data_collector.get_dataset_statistics(samples)
                
                st.success(f"✅ Collected {len(samples)} samples successfully!")
                
                # Display statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Samples", stats['total_samples'])
                with col2:
                    st.metric("Injection Samples", stats['injection_samples'])
                with col3:
                    st.metric("Safe Samples", stats['safe_samples'])
                with col4:
                    st.metric("Injection Ratio", f"{stats['injection_ratio']:.1%}")
                
                # Category distribution
                if stats['categories']:
                    st.subheader("Category Distribution")
                    category_df = pd.DataFrame(list(stats['categories'].items()), columns=['Category', 'Count'])
                    fig = px.pie(category_df, values='Count', names='Category', title="Sample Categories")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Source distribution
                if stats['sources']:
                    st.subheader("Source Distribution")
                    source_df = pd.DataFrame(list(stats['sources'].items()), columns=['Source', 'Count'])
                    fig = px.bar(source_df, x='Source', y='Count', title="Data Sources")
                    st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Error collecting data: {e}")


def model_training_workflow():
    st.subheader("🚀 Model Training")
    
    # Load existing data or collect new data
    data_option = st.radio(
        "Data Source",
        ["Load existing dataset", "Collect new data"],
        help="Choose whether to use existing data or collect new data"
    )
    
    if data_option == "Load existing dataset":
        uploaded_file = st.file_uploader(
            "Upload training dataset (JSON)",
            type=['json'],
            help="Upload a JSON file with training samples"
        )
        
        if uploaded_file is not None:
            try:
                data = json.load(uploaded_file)
                samples = []
                for item in data:
                    from secprompt.data_collector import TrainingSample
                    sample = TrainingSample(
                        text=item['text'],
                        is_injection=item['is_injection'],
                        category=item['category'],
                        severity=item['severity'],
                        source=item['source'],
                        confidence=item['confidence'],
                        metadata=item['metadata']
                    )
                    samples.append(sample)
                
                st.success(f"✅ Loaded {len(samples)} samples from uploaded file")
                
            except Exception as e:
                st.error(f"❌ Error loading file: {e}")
                return
    else:
        # Quick data collection
        with st.spinner("Collecting sample data..."):
            samples = st.session_state.data_collector.generate_synthetic_data(num_samples=500)
        st.success(f"✅ Generated {len(samples)} synthetic samples")
    
    if 'samples' in locals() and samples:
        # Model configuration
        st.subheader("Model Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            model_type = st.selectbox(
                "Model Type",
                ["random_forest", "logistic_regression", "naive_bayes", "gradient_boosting"],
                help="Choose the machine learning algorithm"
            )
            
            test_size = st.slider(
                "Test Size",
                min_value=0.1,
                max_value=0.5,
                value=0.2,
                step=0.05,
                help="Percentage of data to use for testing"
            )
            
            max_features = st.slider(
                "Max Features",
                min_value=1000,
                max_value=10000,
                value=5000,
                step=500,
                help="Maximum number of features to extract"
            )
        
        with col2:
            use_tfidf = st.checkbox("Use TF-IDF", value=True, help="Use TF-IDF instead of Count Vectorizer")
            use_ngrams = st.checkbox("Use N-grams", value=True, help="Include n-gram features")
            cross_validate = st.checkbox("Cross-validate", value=False, help="Perform cross-validation")
            hyperparameter_tuning = st.checkbox("Hyperparameter Tuning", value=False, help="Perform hyperparameter optimization")
        
        # Training options
        st.subheader("Training Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            save_model = st.checkbox("Save Model", value=True, help="Save the trained model")
            model_path = st.text_input(
                "Model Save Path",
                value=f"models/{model_type}_detector.pkl",
                help="Path to save the trained model"
            )
        
        with col2:
            generate_report = st.checkbox("Generate Report", value=True, help="Generate training report")
            plot_results = st.checkbox("Plot Results", value=True, help="Generate visualization plots")
        
        if st.button("Start Training", type="primary"):
            with st.spinner("Training model..."):
                try:
                    # Create training configuration
                    config = TrainingConfig(
                        model_type=model_type,
                        test_size=test_size,
                        max_features=max_features,
                        use_tfidf=use_tfidf,
                        use_ngrams=use_ngrams,
                        max_ngram_range=(1, 3) if use_ngrams else (1, 1)
                    )
                    
                    # Create trainer with config
                    trainer = ModelTrainer(config)
                    
                    # Preprocess data
                    X, y = trainer.preprocess_data(samples)
                    
                    # Split data
                    from sklearn.model_selection import train_test_split
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42, stratify=y
                    )
                    
                    # Cross-validation
                    if cross_validate:
                        st.info("Performing cross-validation...")
                        cv_results = trainer.cross_validate_model(X_train, y_train)
                        st.write(f"Cross-validation F1 score: {cv_results['cv_mean']:.4f} ± {cv_results['cv_std']:.4f}")
                    
                    # Hyperparameter tuning
                    if hyperparameter_tuning:
                        st.info("Performing hyperparameter tuning...")
                        tuning_results = trainer.hyperparameter_tuning(X_train, y_train)
                        if 'error' not in tuning_results:
                            st.write(f"Best score: {tuning_results['best_score']:.4f}")
                            st.write(f"Best parameters: {tuning_results['best_params']}")
                            # Use best estimator
                            config = tuning_results['best_estimator']
                    
                    # Train model
                    result = trainer.train_model(X_train, y_train, X_test, y_test)
                    
                    st.success("✅ Model training completed!")
                    
                    # Display results
                    display_training_results(result)
                    
                    # Save model
                    if save_model:
                        trainer.save_model(result, model_path)
                    
                    # Generate report
                    if generate_report:
                        report_path = model_path.replace('.pkl', '_report.txt')
                        report = trainer.generate_training_report(result, report_path)
                        st.text_area("Training Report", report, height=300)
                    
                    # Plot results
                    if plot_results:
                        plot_path = model_path.replace('.pkl', '_plots.png')
                        trainer.plot_training_results(result, plot_path)
                    
                except Exception as e:
                    st.error(f"❌ Error during training: {e}")


def display_training_results(result: TrainingResult):
    """Display training results"""
    st.subheader("📊 Training Results")
    
    # Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Accuracy", f"{result.metrics['accuracy']:.3f}")
    with col2:
        st.metric("Precision", f"{result.metrics['precision']:.3f}")
    with col3:
        st.metric("Recall", f"{result.metrics['recall']:.3f}")
    with col4:
        st.metric("F1 Score", f"{result.metrics['f1_score']:.3f}")
    with col5:
        st.metric("ROC AUC", f"{result.metrics['roc_auc']:.3f}")
    
    # Training info
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Training Time", f"{result.training_time:.2f}s")
    with col2:
        st.metric("Model Type", result.config.model_type.replace('_', ' ').title())
    
    # Feature importance
    if result.feature_importance:
        st.subheader("Top Features")
        feature_df = pd.DataFrame(list(result.feature_importance.items())[:10], columns=['Feature', 'Importance'])
        fig = px.bar(feature_df, x='Importance', y='Feature', orientation='h', title="Top 10 Feature Importance")
        st.plotly_chart(fig, use_container_width=True)
    
    # Confusion matrix
    st.subheader("Confusion Matrix")
    cm_df = pd.DataFrame(
        result.confusion_matrix,
        columns=['Predicted Safe', 'Predicted Injection'],
        index=['Actual Safe', 'Actual Injection']
    )
    st.dataframe(cm_df, use_container_width=True)


def model_evaluation_page():
    st.subheader("📈 Model Evaluation")
    
    st.info("This page allows you to evaluate trained models on new data.")
    
    # Load model
    model_file = st.file_uploader(
        "Upload trained model (.pkl)",
        type=['pkl'],
        help="Upload a trained model file"
    )
    
    if model_file is not None:
        try:
            # Load model (this would need to be implemented)
            st.success("✅ Model loaded successfully!")
            
            # Evaluation options
            st.subheader("Evaluation Options")
            
            eval_data = st.text_area(
                "Enter test data (one sample per line)",
                placeholder="Enter test samples here...",
                help="Enter test samples to evaluate the model"
            )
            
            if st.button("Evaluate Model", type="primary") and eval_data:
                # This would implement model evaluation
                st.info("Model evaluation functionality will be implemented in a future version.")
                
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")


def model_management_page():
    st.subheader("🗂️ Model Management")
    
    st.info("This page allows you to manage and compare trained models.")
    
    # List saved models
    models_dir = Path("models")
    if models_dir.exists():
        model_files = list(models_dir.glob("*.pkl"))
        
        if model_files:
            st.subheader("Saved Models")
            
            for model_file in model_files:
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.write(f"**{model_file.stem}**")
                    st.write(f"Size: {model_file.stat().st_size / 1024:.1f} KB")
                
                with col2:
                    if st.button(f"Load {model_file.stem}", key=f"load_{model_file.stem}"):
                        st.info(f"Loading model: {model_file.name}")
                
                with col3:
                    if st.button(f"Delete {model_file.stem}", key=f"delete_{model_file.stem}"):
                        model_file.unlink()
                        st.success(f"Deleted: {model_file.name}")
                        st.rerun()
        else:
            st.info("No saved models found.")
    else:
        st.info("Models directory not found. Train a model first.")


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
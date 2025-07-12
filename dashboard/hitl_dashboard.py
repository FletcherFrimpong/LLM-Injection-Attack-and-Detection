"""
Human-in-the-Loop Security Dashboard

A Streamlit dashboard for managing HITL security workflows with SecPrompt.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import time
from typing import Dict, List

# Import SecPrompt components
from secprompt.detector import PromptDetector
from secprompt.evaluator import PromptEvaluator
from secprompt.defenses import PromptDefender
from hitl_integration import HITLSecuritySystem, HITLAction


def main():
    st.set_page_config(
        page_title="SecPrompt HITL Dashboard",
        page_icon="🛡️",
        layout="wide"
    )
    
    st.title("🛡️ SecPrompt Human-in-the-Loop Security Dashboard")
    st.markdown("Manage AI security with human oversight and automated detection")
    
    # Initialize session state
    if 'hitl_system' not in st.session_state:
        st.session_state.hitl_system = HITLSecuritySystem()
    
    if 'requests' not in st.session_state:
        st.session_state.requests = []
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Thresholds
        st.subheader("Decision Thresholds")
        auto_allow = st.slider("Auto Allow Threshold", 0.0, 1.0, 0.9, 0.1)
        human_review = st.slider("Human Review Threshold", 0.0, 1.0, 0.7, 0.1)
        escalation = st.slider("Escalation Threshold", 0.0, 1.0, 0.3, 0.1)
        
        # Environment settings
        st.subheader("Environment")
        production_mode = st.checkbox("Production Environment", value=True)
        
        # Update system thresholds
        st.session_state.hitl_system.auto_allow_threshold = auto_allow
        st.session_state.hitl_system.human_review_threshold = human_review
        st.session_state.hitl_system.escalation_threshold = escalation
        
        # Quick actions
        st.subheader("Quick Actions")
        if st.button("Clear All Data"):
            st.session_state.hitl_system = HITLSecuritySystem()
            st.session_state.requests = []
            st.rerun()
        
        if st.button("Export Data"):
            export_data()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🔍 Input Testing")
        
        # Input form
        with st.form("input_test"):
            user_input = st.text_area(
                "Enter text to test:",
                placeholder="Type your text here...",
                height=100
            )
            
            col1, col2 = st.columns(2)
            with col1:
                user_id = st.text_input("User ID", value="test_user")
            with col2:
                session_id = st.text_input("Session ID", value="test_session")
            
            submitted = st.form_submit_button("Test Input")
            
            if submitted and user_input:
                process_input(user_input, user_id, session_id, production_mode)
    
    with col2:
        st.header("📊 Real-time Stats")
        display_statistics()
    
    # Review Queue
    st.header("👥 Human Review Queue")
    display_review_queue()
    
    # Historical Data
    st.header("📈 Historical Analysis")
    display_historical_data()


def process_input(user_input: str, user_id: str, session_id: str, production_mode: bool):
    """Process user input through HITL system"""
    
    context = {"production_environment": production_mode}
    user_info = {"user_id": user_id, "session_id": session_id}
    
    # Process through HITL system
    result = st.session_state.hitl_system.process_input(
        user_input, user_info, context
    )
    
    # Store request
    st.session_state.requests.append({
        "timestamp": datetime.now(),
        "input": user_input,
        "result": result
    })
    
    # Display results
    st.success("✅ Input processed successfully!")
    
    # Show results in expandable section
    with st.expander("📋 Processing Results", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Action", result['action'].upper())
            st.metric("Confidence", f"{result['confidence']:.2%}")
            st.metric("Severity", result['severity'].upper())
        
        with col2:
            st.metric("Human Review", "Required" if result['requires_human_review'] else "Not Required")
            st.metric("Request ID", result['request_id'])
            st.metric("Timestamp", result['timestamp'])
        
        st.subheader("Reasoning")
        st.info(result['reasoning'])
        
        if result['action'] == 'sanitize':
            st.subheader("Defended Input")
            st.code(result['defended_input'])


def display_statistics():
    """Display real-time statistics"""
    
    stats = st.session_state.hitl_system.get_statistics()
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Requests", stats['total_requests'])
    
    with col2:
        st.metric("Auto Allowed", f"{stats['auto_allow_rate']:.1%}")
    
    with col3:
        st.metric("Human Reviews", f"{stats['human_review_rate']:.1%}")
    
    with col4:
        st.metric("Escalations", f"{stats['escalation_rate']:.1%}")
    
    # Pie chart of actions
    if stats['total_requests'] > 0:
        actions_data = {
            'Auto Allowed': stats['auto_allowed'],
            'Auto Blocked': stats['auto_blocked'],
            'Human Reviews': stats['human_reviews'],
            'Escalations': stats['escalations']
        }
        
        fig = px.pie(
            values=list(actions_data.values()),
            names=list(actions_data.keys()),
            title="Action Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)


def display_review_queue():
    """Display pending human reviews"""
    
    queue = st.session_state.hitl_system.export_review_queue()
    
    if not queue:
        st.info("No pending reviews")
        return
    
    # Create DataFrame for better display
    df = pd.DataFrame(queue)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Display each review item
    for _, item in df.iterrows():
        with st.expander(f"Review {item['request_id']} - {item['severity'].upper()}", expanded=False):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.text_area("Input Text", item['user_input'], height=100, disabled=True)
                st.info(f"Confidence: {item['confidence']:.2%}")
            
            with col2:
                st.metric("Severity", item['severity'].upper())
                st.metric("Recommended", item['recommended_action'].upper())
                
                # Action buttons
                if st.button(f"Allow", key=f"allow_{item['request_id']}"):
                    handle_human_decision(item['request_id'], "allow")
                
                if st.button(f"Block", key=f"block_{item['request_id']}"):
                    handle_human_decision(item['request_id'], "block")
                
                if st.button(f"Sanitize", key=f"sanitize_{item['request_id']}"):
                    handle_human_decision(item['request_id'], "sanitize")


def handle_human_decision(request_id: str, action: str):
    """Handle human decision on review item"""
    st.success(f"Human decision recorded: {action.upper()} for {request_id}")
    # In a real system, this would update the database and trigger actions


def display_historical_data():
    """Display historical analysis"""
    
    if not st.session_state.requests:
        st.info("No historical data available")
        return
    
    # Create DataFrame
    df = pd.DataFrame([
        {
            'timestamp': req['timestamp'],
            'input': req['input'][:50] + "..." if len(req['input']) > 50 else req['input'],
            'action': req['result']['action'],
            'confidence': req['result']['confidence'],
            'severity': req['result']['severity'],
            'human_review': req['result']['requires_human_review']
        }
        for req in st.session_state.requests
    ])
    
    # Time series of actions
    fig = px.line(
        df,
        x='timestamp',
        y='confidence',
        color='action',
        title="Confidence Over Time"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Severity distribution
    severity_counts = df['severity'].value_counts()
    fig = px.bar(
        x=severity_counts.index,
        y=severity_counts.values,
        title="Severity Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Raw data table
    st.subheader("Raw Data")
    st.dataframe(df, use_container_width=True)


def export_data():
    """Export data to JSON"""
    
    data = {
        "statistics": st.session_state.hitl_system.get_statistics(),
        "requests": st.session_state.requests,
        "export_timestamp": datetime.now().isoformat()
    }
    
    # Create download button
    st.download_button(
        label="Download Data",
        data=json.dumps(data, indent=2, default=str),
        file_name=f"secprompt_hitl_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )


if __name__ == "__main__":
    main() 
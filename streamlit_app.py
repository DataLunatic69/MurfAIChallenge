# Professional Debate System - Streamlit UI
# streamlit_app.py

import streamlit as st
import streamlit.components.v1 as components
from datetime import datetime
import json
import time
import threading
from typing import Dict, List, Optional
import os
import base64
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Import the debate system with proper error handling
try:
    from run import (
        enhanced_debate_app, 
        DebateState, 
        SpeakerType,
        DebatePhase,
        audio_streamer,
        speak_text_streaming,
        speech_to_text_whisper_api,
        CONFIG,
        MURF_CONFIG,
        cleanup_audio
    )
    DEBATE_SYSTEM_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Debate system not fully available: {e}")
    DEBATE_SYSTEM_AVAILABLE = False

# Import the debate integration
try:
    from debate_integration import DebateSystemIntegration, debate_integration
    DEBATE_INTEGRATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Debate integration not available: {e}")
    DEBATE_INTEGRATION_AVAILABLE = False
    # Create a mock class for fallback
    class DebateSystemIntegration:
        def __init__(self):
            self.is_running = False
        
        def start_debate(self, topic, config):
            return {"success": True, "positions": {"user_position": "For", "ai_position": "Against"}}
        
        def submit_user_argument(self, argument):
            return {
                "success": True, 
                "user_score": 30, 
                "user_feedback": "Good argument",
                "ai_response": "This is a simulated AI response.",
                "ai_score": 28,
                "ai_feedback": "Standard response",
                "round_complete": True
            }
    
    debate_integration = DebateSystemIntegration()

# Page Configuration
st.set_page_config(
    page_title="Murf Debate Chamber",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional Courtroom Aesthetic
def load_custom_css():
    st.markdown("""
    <style>
    /* Import classical fonts */
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Crimson+Text:wght@400;600&family=Libre+Baskerville:wght@400;700&display=swap');
    
    /* Main background with elegant gradient */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        font-family: 'Libre Baskerville', serif;
    }
    
    /* Header styling with elegant typography */
    .main-header {
        font-family: 'Playfair Display', serif;
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        color: #2c3e50;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .sub-header {
        font-family: 'Crimson Text', serif;
        font-size: 1.3rem;
        text-align: center;
        color: #7f8c8d;
        margin-bottom: 2rem;
        font-style: italic;
    }
    
    /* Elegant card styling */
    .elegant-card {
        background: white;
        border-radius: 15px;
        padding: 25px;
        margin: 15px 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .elegant-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0,0,0,0.15);
    }
    
    .elegant-card h4 {
        color: #2c3e50;
        font-family: 'Playfair Display', serif;
        margin-bottom: 15px;
        font-size: 1.3rem;
    }
    
    .elegant-card p {
        color: #2c3e50;
        font-family: 'Crimson Text', serif;
        line-height: 1.6;
        font-size: 1.05rem;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 14px 28px;
        font-family: 'Crimson Text', serif;
        font-size: 1.1rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.25);
    }
    
    /* Secondary button */
    .secondary-button {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%) !important;
    }
    
    /* Text area styling */
    .stTextArea textarea {
        background: #f8f9fa;
        border: 2px solid #e9ecef;
        border-radius: 10px;
        padding: 15px;
        font-family: 'Crimson Text', serif;
        font-size: 1.05rem;
        color: #2c3e50;
        transition: border-color 0.3s ease;
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2);
    }
    
    .stTextArea textarea::placeholder {
        color: #7f8c8d;
        opacity: 0.8;
    }
    
    /* Score displays */
    .score-display {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .score-value {
        font-size: 2.8rem;
        font-weight: 700;
        font-family: 'Playfair Display', serif;
    }
    
    .score-label {
        font-size: 1.1rem;
        opacity: 0.9;
        font-family: 'Crimson Text', serif;
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: white;
        border-radius: 8px 8px 0 0;
        padding: 12px 24px;
        font-family: 'Crimson Text', serif;
        font-weight: 600;
        border: 1px solid #e0e0e0;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Metric styling */
    [data-testid="metric-container"] {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
    }
    
    /* Sidebar text styling */
    .css-1d391kg .stMarkdown h1,
    .css-1d391kg .stMarkdown h2,
    .css-1d391kg .stMarkdown h3,
    .css-1d391kg .stMarkdown h4 {
        color: #ecf0f1 !important;
    }
    
    .css-1d391kg .stMarkdown p,
    .css-1d391kg .stMarkdown div {
        color: #ecf0f1 !important;
    }
    
    /* Sidebar input styling */
    .css-1d391kg .stTextInput input,
    .css-1d391kg .stTextArea textarea {
        background: #34495e !important;
        border: 2px solid #5d6d7e !important;
        color: #ecf0f1 !important;
        border-radius: 8px;
    }
    
    .css-1d391kg .stTextInput input:focus,
    .css-1d391kg .stTextArea textarea:focus {
        border-color: #3498db !important;
        box-shadow: 0 0 0 2px rgba(52, 152, 219, 0.2) !important;
    }
    
    .css-1d391kg .stTextInput input::placeholder,
    .css-1d391kg .stTextArea textarea::placeholder {
        color: #bdc3c7 !important;
    }
    
    /* Sidebar selectbox styling */
    .css-1d391kg .stSelectbox > div > div {
        background: #34495e !important;
        border: 2px solid #5d6d7e !important;
        color: #ecf0f1 !important;
    }
    
    /* Sidebar slider styling */
    .css-1d391kg .stSlider > div > div > div > div {
        background: #3498db !important;
    }
    
    .css-1d391kg .stSlider > div > div > div > div > div {
        background: #ecf0f1 !important;
        border: 2px solid #3498db !important;
    }
    
    /* Sidebar checkbox styling */
    .css-1d391kg .stCheckbox > div > div {
        background: #34495e !important;
        border: 2px solid #5d6d7e !important;
    }
    
    .css-1d391kg .stCheckbox > div > div[data-testid="stCheckbox"] {
        color: #ecf0f1 !important;
    }
    
    /* Sidebar button styling */
    .css-1d391kg .stButton > button {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%) !important;
        color: #ecf0f1 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px 24px !important;
        font-family: 'Crimson Text', serif !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2) !important;
        transition: all 0.3s ease !important;
    }
    
    .css-1d391kg .stButton > button:hover {
        background: linear-gradient(135deg, #2980b9 0%, #3498db 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(0,0,0,0.25) !important;
    }
    
    /* Sidebar secondary button styling */
    .css-1d391kg .stButton > button[data-baseweb="button"] {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%) !important;
    }
    
    .css-1d391kg .stButton > button[data-baseweb="button"]:hover {
        background: linear-gradient(135deg, #c0392b 0%, #e74c3c 100%) !important;
    }
    
    /* Sidebar label styling */
    .css-1d391kg label {
        color: #ecf0f1 !important;
        font-weight: 600 !important;
    }
    
    /* Sidebar help text styling */
    .css-1d391kg .stMarkdown small {
        color: #bdc3c7 !important;
    }
    
    /* Argument cards */
    .argument-card {
        background: white;
        border-left: 4px solid #667eea;
        border-radius: 8px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .argument-card.ai {
        border-left-color: #f5576c;
    }
    
    .argument-card h4 {
        color: #2c3e50;
        font-family: 'Playfair Display', serif;
        margin-bottom: 10px;
        font-size: 1.2rem;
    }
    
    .argument-card p {
        color: #7f8c8d;
        font-family: 'Crimson Text', serif;
        line-height: 1.6;
        font-size: 1.05rem;
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        margin: 5px 0;
    }
    
    .status-active {
        background: #4CAF50;
        color: white;
    }
    
    .status-waiting {
        background: #FF9800;
        color: white;
    }
    
    .status-inactive {
        background: #9E9E9E;
        color: white;
    }
    
    /* Feature highlights */
    .feature-card {
        text-align: center;
        padding: 25px;
        border-radius: 15px;
        background: white;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
    }
    
    .feature-card h4 {
        color: #2c3e50;
        font-family: 'Playfair Display', serif;
        margin-bottom: 15px;
        font-size: 1.3rem;
    }
    
    .feature-card p {
        color: #2c3e50;
        font-family: 'Crimson Text', serif;
        line-height: 1.6;
        font-size: 1.05rem;
    }
    
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 15px;
        color: #667eea;
    }
    
    /* Welcome section */
    .welcome-section {
        text-align: center;
        padding: 40px;
        background: white;
        border-radius: 20px;
        box-shadow: 0 15px 50px rgba(0,0,0,0.1);
        margin: 20px 0;
    }
    
    /* Ensure all text in white cards is readable */
    .stMarkdown div[data-testid="stMarkdownContainer"] {
        color: #2c3e50;
    }
    
    /* Override any light text colors in cards */
    .stMarkdown div[data-testid="stMarkdownContainer"] h4,
    .stMarkdown div[data-testid="stMarkdownContainer"] p {
        color: #2c3e50 !important;
    }
    
    /* Ensure input text is visible */
    .stTextInput input,
    .stTextArea textarea {
        color: #2c3e50 !important;
    }
    
    /* Ensure labels are visible */
    .stTextInput label,
    .stTextArea label {
        color: #2c3e50 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    if 'debate_state' not in st.session_state:
        st.session_state.debate_state = None
    if 'debate_history' not in st.session_state:
        st.session_state.debate_history = []
    if 'current_round' not in st.session_state:
        st.session_state.current_round = 1
    if 'debate_active' not in st.session_state:
        st.session_state.debate_active = False
    if 'audio_enabled' not in st.session_state:
        st.session_state.audio_enabled = False
    if 'voice_input_enabled' not in st.session_state:
        st.session_state.voice_input_enabled = False
    if 'user_argument' not in st.session_state:
        st.session_state.user_argument = ""
    if 'debate_transcript' not in st.session_state:
        st.session_state.debate_transcript = []
    if 'current_topic' not in st.session_state:
        st.session_state.current_topic = ""
    if 'user_position' not in st.session_state:
        st.session_state.user_position = ""
    if 'ai_position' not in st.session_state:
        st.session_state.ai_position = ""
    if 'user_score' not in st.session_state:
        st.session_state.user_score = 0
    if 'ai_score' not in st.session_state:
        st.session_state.ai_score = 0
    if 'debate_config' not in st.session_state:
        st.session_state.debate_config = {"max_rounds": 3, "time_limit": 120}
    if 'current_speaker' not in st.session_state:
        st.session_state.current_speaker = None
    if 'waiting_for_ai' not in st.session_state:
        st.session_state.waiting_for_ai = False
    if 'last_ai_argument' not in st.session_state:
        st.session_state.last_ai_argument = ""
    if 'last_judge_feedback' not in st.session_state:
        st.session_state.last_judge_feedback = ""
    if 'transcribed_text' not in st.session_state:
        st.session_state.transcribed_text = ""
    if 'user_argument_text' not in st.session_state:
        st.session_state.user_argument_text = ""
    if 'debate_integration' not in st.session_state:
        st.session_state.debate_integration = debate_integration if DEBATE_INTEGRATION_AVAILABLE else DebateSystemIntegration()
    
    # API Keys
    if 'openai_api_key' not in st.session_state:
        st.session_state.openai_api_key = ""
    if 'groq_api_key' not in st.session_state:
        st.session_state.groq_api_key = ""
    if 'murf_api_key' not in st.session_state:
        st.session_state.murf_api_key = ""

# Update configuration with API keys
def update_config_with_api_keys():
    """Update the configuration objects with session state API keys."""
    if DEBATE_SYSTEM_AVAILABLE:
        if st.session_state.openai_api_key.strip():
            CONFIG["openai_api_key"] = st.session_state.openai_api_key
        if st.session_state.groq_api_key.strip():
            CONFIG["groq_api_key"] = st.session_state.groq_api_key
        if st.session_state.murf_api_key.strip():
            MURF_CONFIG["api_key"] = st.session_state.murf_api_key

# Header with elegant design
def render_header():
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.markdown("""
        <div class="main-header">
            ⚖️ Murf Debate Chamber
        </div>
        <div class="sub-header">
            A Distinguished Platform for Intellectual Discourse and Rhetorical Excellence
        </div>
        """, unsafe_allow_html=True)

# Sidebar configuration with improved layout
def render_sidebar():
    with st.sidebar:
        # API Keys Section
        st.markdown("## 🔑 API Configuration")
        st.markdown("---")
        
        openai_key = st.text_input(
            "OpenAI API Key:",
            value=st.session_state.openai_api_key,
            type="password",
            placeholder="sk-...",
            help="Enter your OpenAI API key for GPT models"
        )
        if openai_key != st.session_state.openai_api_key:
            st.session_state.openai_api_key = openai_key
        
        groq_key = st.text_input(
            "Groq API Key:",
            value=st.session_state.groq_api_key,
            type="password",
            placeholder="gsk_...",
            help="Enter your Groq API key for fast inference"
        )
        if groq_key != st.session_state.groq_api_key:
            st.session_state.groq_api_key = groq_key
        
        murf_key = st.text_input(
            "Murf AI API Key:",
            value=st.session_state.murf_api_key,
            type="password",
            placeholder="Enter your Murf AI key",
            help="Enter your Murf AI API key for voice synthesis"
        )
        if murf_key != st.session_state.murf_api_key:
            st.session_state.murf_api_key = murf_key
        
        # Update configuration when API keys change
        if (openai_key != st.session_state.openai_api_key or 
            groq_key != st.session_state.groq_api_key or 
            murf_key != st.session_state.murf_api_key):
            update_config_with_api_keys()
        
        # API Key Status
        st.markdown("### 🔍 API Status")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            status_color = "🟢" if st.session_state.openai_api_key.strip() else "🔴"
            st.markdown(f"{status_color} OpenAI")
        
        with col2:
            status_color = "🟢" if st.session_state.groq_api_key.strip() else "🔴"
            st.markdown(f"{status_color} Groq")
        
        with col3:
            status_color = "🟢" if st.session_state.murf_api_key.strip() else "🔴"
            st.markdown(f"{status_color} Murf AI")
        
        st.markdown("---")
        
        st.markdown("## ⚙️ Debate Configuration")
        st.markdown("---")
        
        # Topic selection
        st.markdown("### 📜 Debate Topic")
        preset_topics = [
            "Custom Topic",
            "Should artificial intelligence be regulated by the government?",
            "Is artificial general intelligence an existential threat to humanity?",
            "Should governments implement universal basic income?",
            "Is remote work better than office work for productivity?",
            "Should social media platforms be held responsible for user-generated content?"
        ]
        
        selected_preset = st.selectbox(
            "Select a debate topic:",
            preset_topics,
            key="topic_preset"
        )
        
        if selected_preset == "Custom Topic":
            topic = st.text_area(
                "Enter your custom debate topic:",
                height=100,
                key="custom_topic",
                placeholder="e.g., Should renewable energy replace fossil fuels completely?"
            )
        else:
            topic = selected_preset
            
        st.markdown("---")
        
        # Debate settings
        st.markdown("### ⚡ Debate Parameters")
        
        col1, col2 = st.columns(2)
        with col1:
            max_rounds = st.slider(
                "Number of Rounds:",
                min_value=1,
                max_value=5,
                value=st.session_state.debate_config["max_rounds"],
                key="max_rounds_slider"
            )
        
        with col2:
            time_limit = st.slider(
                "Time per Argument (s):",
                min_value=30,
                max_value=300,
                value=st.session_state.debate_config["time_limit"],
                step=30,
                key="time_limit_slider"
            )
        
        st.session_state.debate_config["max_rounds"] = max_rounds
        st.session_state.debate_config["time_limit"] = time_limit
        
        st.markdown("---")
        
        # Audio settings
        st.markdown("### 🔊 Audio Settings")
        
        # Check if API keys are available
        has_murf_key = bool(st.session_state.murf_api_key.strip())
        has_openai_key = bool(st.session_state.openai_api_key.strip())
        
        audio_enabled = st.checkbox(
            "Enable Voice Output",
            value=st.session_state.audio_enabled,
            disabled=not has_murf_key,
            help="Enable AI voice responses (requires Murf AI API key)"
        )
        st.session_state.audio_enabled = audio_enabled
        
        voice_input = st.checkbox(
            "Enable Voice Input",
            value=st.session_state.voice_input_enabled,
            disabled=not has_openai_key,
            help="Enable voice-to-text for your arguments (requires OpenAI API key)"
        )
        st.session_state.voice_input_enabled = voice_input
        
        if audio_enabled:
            voice_speed = st.slider(
                "Speech Speed:",
                min_value=0.5,
                max_value=2.0,
                value=1.0,
                step=0.1,
                key="voice_speed"
            )
            if DEBATE_SYSTEM_AVAILABLE:
                MURF_CONFIG["speed"] = voice_speed
        
        st.markdown("---")
        
        # Action buttons
        if not st.session_state.debate_active:
            if st.button("🎯 **COMMENCE DEBATE**", use_container_width=True, type="primary"):
                if topic and topic != "Custom Topic":
                    start_debate(topic, max_rounds)
                else:
                    st.error("Please select or enter a valid debate topic.")
        else:
            if st.button("⛔ **ADJOURN DEBATE**", use_container_width=True, type="secondary"):
                stop_debate()
        
        st.markdown("---")
        
        # Export options
        st.markdown("### 📁 Export Data")
        
        col_export1, col_export2 = st.columns(2)
        with col_export1:
            if st.button("💾 Save Transcript", use_container_width=True):
                save_transcript()
        with col_export2:
            if st.button("📊 Export Stats", use_container_width=True):
                export_statistics()

# Main debate interface
def render_main_interface():
    if not st.session_state.debate_active:
        render_welcome_screen()
    else:
        render_debate_screen()

# Welcome screen with feature highlights
def render_welcome_screen():
    st.markdown("""
    <div class="welcome-section">
        <h2 style='font-family: "Playfair Display", serif; color: #2c3e50;'>
            Welcome to the Murf Debate Chamber
        </h2>
        <p style='font-family: "Crimson Text", serif; color: #7f8c8d; font-size: 1.2rem;'>
            Engage in sophisticated intellectual discourse with AI-powered opponents and impartial adjudication.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature highlights
    st.markdown("### 🎯 Key Features")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🎙️</div>
            <h4>Voice Integration</h4>
            <p>Natural voice input and AI vocal responses for immersive debate experience</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">⚖️</div>
            <h4>Impartial Judging</h4>
            <p>AI-powered evaluation based on logic, evidence, and rhetorical effectiveness</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <h4>Detailed Analytics</h4>
            <p>Comprehensive scoring, feedback, and performance metrics for improvement</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick start guide
    st.markdown("---")
    st.markdown("### 🚀 Quick Start Guide")
    
    guide_col1, guide_col2, guide_col3 = st.columns(3)
    
    with guide_col1:
        st.markdown("""
        <div class="elegant-card">
            <h4>1. Configure</h4>
            <p>Select your debate topic and adjust parameters in the sidebar</p>
        </div>
        """, unsafe_allow_html=True)
    
    with guide_col2:
        st.markdown("""
        <div class="elegant-card">
            <h4>2. Commence</h4>
            <p>Click "Commence Debate" to start your intellectual engagement</p>
        </div>
        """, unsafe_allow_html=True)
    
    with guide_col3:
        st.markdown("""
        <div class="elegant-card">
            <h4>3. Debate</h4>
            <p>Present your arguments and respond to AI counterpoints</p>
        </div>
        """, unsafe_allow_html=True)
    
    # System status
    st.markdown("---")
    st.markdown("### 🔧 System Status")
    
    status_col1, status_col2 = st.columns(2)
    
    with status_col1:
        status = "🟢 Available" if DEBATE_SYSTEM_AVAILABLE else "🔴 Unavailable"
        st.markdown(f"**Debate System:** {status}")
        
        audio_status = "🟢 Enabled" if st.session_state.audio_enabled else "⚪ Disabled"
        st.markdown(f"**Voice Output:** {audio_status}")
    
    with status_col2:
        integration_status = "🟢 Available" if DEBATE_INTEGRATION_AVAILABLE else "🟡 Simulation Mode"
        st.markdown(f"**AI Integration:** {integration_status}")
        
        voice_input_status = "🟢 Enabled" if st.session_state.voice_input_enabled else "⚪ Disabled"
        st.markdown(f"**Voice Input:** {voice_input_status}")

# Active debate screen
def render_debate_screen():
    # Debate header with status
    col_header1, col_header2, col_header3 = st.columns([2, 1, 2])
    
    with col_header1:
        st.markdown(f"""
        <div class="elegant-card">
            <h3>Current Debate</h3>
            <p style="color: #667eea; font-size: 1.2rem;">{st.session_state.current_topic}</p>
            <span class="status-indicator status-active">Round {st.session_state.current_round} of {st.session_state.debate_config['max_rounds']}</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col_header2:
        speaker_status = "🎤 Your Turn" if st.session_state.current_speaker == 'user' else "🤖 AI's Turn"
        status_class = "status-active" if st.session_state.current_speaker == 'user' else "status-waiting"
        st.markdown(f"""
        <div class="elegant-card" style="text-align: center;">
            <h4>Current Speaker</h4>
            <span class="status-indicator {status_class}">{speaker_status}</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col_header3:
        st.markdown(f"""
        <div class="elegant-card">
            <h4>Time Remaining</h4>
            <div style="text-align: center; font-size: 2rem; color: #667eea;">
                {st.session_state.debate_config['time_limit']}s
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Score dashboard
    st.markdown("---")
    st.markdown("### 📊 Score Dashboard")
    
    col_score1, col_score2, col_score3, col_score4 = st.columns(4)
    
    with col_score1:
        st.markdown(f"""
        <div class="score-display">
            <div class="score-label">USER SCORE</div>
            <div class="score-value">{st.session_state.user_score}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_score2:
        st.markdown(f"""
        <div class="score-display">
            <div class="score-label">AI SCORE</div>
            <div class="score-value">{st.session_state.ai_score}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_score3:
        progress = min(st.session_state.current_round / st.session_state.debate_config["max_rounds"], 1.0)
        st.markdown(f"""
        <div class="elegant-card">
            <h4>Round Progress</h4>
            <div style="text-align: center;">
                {st.session_state.current_round} / {st.session_state.debate_config['max_rounds']}
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(progress)
    
    with col_score4:
        avg_user = calculate_average_score('user')
        avg_ai = calculate_average_score('ai')
        leader = "👤 You" if st.session_state.user_score > st.session_state.ai_score else "🤖 AI" if st.session_state.ai_score > st.session_state.user_score else "⚖️ Tie"
        st.markdown(f"""
        <div class="elegant-card">
            <h4>Current Leader</h4>
            <div style="text-align: center; font-size: 1.5rem; color: #667eea;">
                {leader}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Debate content area
    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["🎭 **Live Debate**", "📜 **Transcript**", "📊 **Analysis**"])
    
    with tab1:
        render_live_debate()
    
    with tab2:
        render_transcript()
    
    with tab3:
        render_analysis()

# Live debate tab
def render_live_debate():
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 👤 Your Position")
        st.markdown(f"""
        <div class="elegant-card">
            <p style="font-size: 1.1rem; color: #2c3e50;">{st.session_state.user_position}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 💬 Your Argument")
        
        if st.session_state.current_speaker == 'user' and not st.session_state.waiting_for_ai:
            with st.form(key="argument_form"):
                user_argument = st.text_area(
                    "Present your argument:",
                    height=150,
                    key="user_argument_text",
                    value=st.session_state.get('transcribed_text', st.session_state.get('user_argument_text', '')),
                    placeholder="Type your argument here... (Minimum 50 characters)"
                )
                
                col_a, col_b = st.columns([2, 1])
                with col_a:
                    submit_button = st.form_submit_button("🚀 **SUBMIT ARGUMENT**", use_container_width=True)
                with col_b:
                    if st.session_state.voice_input_enabled:
                        voice_button = st.form_submit_button("🎤 **VOICE INPUT**", use_container_width=True, type="secondary")
                    else:
                        voice_button = False
                
                if submit_button and user_argument:
                    submit_user_argument(user_argument)
                    if 'transcribed_text' in st.session_state:
                        st.session_state.transcribed_text = ""
                elif voice_button:
                    record_voice_input()
        else:
            st.markdown("""
            <div class="elegant-card">
                <p style="text-align: center; color: #7f8c8d;">
                    ⏳ Waiting for your turn to speak...
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🤖 AI Position")
        st.markdown(f"""
        <div class="elegant-card">
            <p style="font-size: 1.1rem; color: #2c3e50;">{st.session_state.ai_position}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 💭 AI's Response")
        
        if st.session_state.last_ai_argument:
            st.markdown(f"""
            <div class="argument-card ai">
                <h4>🤖 AI Argument</h4>
                <p>{st.session_state.last_ai_argument}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="elegant-card">
                <p style="text-align: center; color: #7f8c8d;">
                    ⏳ AI is preparing a response...
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    # Judge feedback section
    st.markdown("---")
    st.markdown("### ⚖️ Judge's Evaluation")
    
    if st.session_state.last_judge_feedback:
        st.markdown(f"""
        <div class="elegant-card" style="background: linear-gradient(135deg, #fdfcfb 0%, #e2d1c3 100%);">
            <h4>🏆 Evaluation Results</h4>
            <p style="font-size: 1.1rem; line-height: 1.6;">{st.session_state.last_judge_feedback}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="elegant-card">
            <p style="text-align: center; color: #7f8c8d;">
                📝 Awaiting arguments for evaluation...
            </p>
        </div>
        """, unsafe_allow_html=True)

# Transcript tab
def render_transcript():
    if st.session_state.debate_transcript:
        for entry in st.session_state.debate_transcript:
            card_class = "argument-card" if entry['speaker'] == 'user' else "argument-card ai"
            speaker_icon = "👤" if entry['speaker'] == 'user' else "🤖"
            
            st.markdown(f"""
            <div class="{card_class}">
                <h4>{speaker_icon} {entry['speaker'].upper()} - Round {entry['round']}</h4>
                <p>{entry['content']}</p>
                <div style="margin-top: 10px;">
                    <small style="color: #667eea;">Score: {entry.get('score', 'Pending')}/40</small>
                    <small style="color: #7f8c8d; margin-left: 15px;">
                        {datetime.fromisoformat(entry['timestamp']).strftime('%H:%M:%S')}
                    </small>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="elegant-card">
            <p style="text-align: center; color: #7f8c8d;">
                📄 No arguments yet. The transcript will appear here as the debate progresses.
            </p>
        </div>
        """, unsafe_allow_html=True)

# Analysis tab
def render_analysis():
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Performance Analytics")
        
        if len(st.session_state.debate_transcript) > 0:
            rounds = []
            user_scores = []
            ai_scores = []
            
            for entry in st.session_state.debate_transcript:
                if entry.get('score'):
                    rounds.append(f"Round {entry['round']}")
                    if entry['speaker'] == 'user':
                        user_scores.append(entry['score'])
                    else:
                        ai_scores.append(entry['score'])
            
            if rounds:
                chart_data = {
                    "Round": rounds,
                    "User Score": user_scores,
                    "AI Score": ai_scores
                }
                st.line_chart(chart_data, x="Round", y=["User Score", "AI Score"])
        else:
            st.markdown("""
            <div class="elegant-card">
                <p style="text-align: center; color: #7f8c8d;">
                    📊 Performance data will appear after scored rounds.
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🎯 Key Metrics")
        
        col_metric1, col_metric2 = st.columns(2)
        
        with col_metric1:
            avg_user = calculate_average_score('user')
            st.metric("Avg. User Score", f"{avg_user:.1f}")
            st.metric("Total Rounds", st.session_state.current_round - 1)
        
        with col_metric2:
            avg_ai = calculate_average_score('ai')
            st.metric("Avg. AI Score", f"{avg_ai:.1f}")
            st.metric("Score Difference", abs(st.session_state.user_score - st.session_state.ai_score))
    
    st.markdown("---")
    st.markdown("### 💡 Improvement Suggestions")
    
    suggestions = generate_improvement_suggestions()
    if suggestions:
        for suggestion in suggestions:
            st.markdown(f"""
            <div class="elegant-card" style="background: #fff3cd; border-left: 4px solid #ffc107;">
                <p>💡 {suggestion}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="elegant-card">
            <p style="text-align: center; color: #7f8c8d;">
                💡 Suggestions will appear based on your performance.
            </p>
        </div>
        """, unsafe_allow_html=True)

# Helper functions (keep the same as before)
def start_debate(topic: str, max_rounds: int):
    """Initialize and start a new debate."""
    st.session_state.debate_active = True
    st.session_state.current_topic = topic
    st.session_state.current_round = 1
    st.session_state.user_score = 0
    st.session_state.ai_score = 0
    st.session_state.debate_transcript = []
    st.session_state.current_speaker = 'user'
    st.session_state.waiting_for_ai = False
    
    # Clear any previous transcribed text
    st.session_state.transcribed_text = ""
    st.session_state.user_argument_text = ""
    
    # Initialize debate using the integration layer
    result = st.session_state.debate_integration.start_debate(
        topic, 
        {"max_rounds": max_rounds, "time_limit": st.session_state.debate_config["time_limit"]}
    )
    
    if result["success"]:
        st.session_state.user_position = result["positions"]["user_position"]
        st.session_state.ai_position = result["positions"]["ai_position"]
        st.success(f"Debate commenced! Topic: {topic}")
    else:
        st.error(f"Failed to start debate: {result.get('error', 'Unknown error')}")
        st.session_state.debate_active = False
    
    st.rerun()

def stop_debate():
    """Stop the current debate."""
    st.session_state.debate_active = False
    if DEBATE_SYSTEM_AVAILABLE:
        audio_streamer.stop()
    
    # Clear debate-related session state
    st.session_state.transcribed_text = ""
    st.session_state.user_argument_text = ""
    
    st.info("Debate has been adjourned.")
    st.rerun()

def submit_user_argument(argument: str):
    """Process user argument submission."""
    if argument and len(argument) >= 50:
        # Add to transcript
        st.session_state.debate_transcript.append({
            'speaker': 'user',
            'round': st.session_state.current_round,
            'content': argument,
            'timestamp': datetime.now().isoformat(),
            'score': None  # Will be updated after judging
        })
        
        # Switch to AI turn
        st.session_state.current_speaker = 'ai'
        st.session_state.waiting_for_ai = True
        
        # Use the debate integration to process the argument
        result = st.session_state.debate_integration.submit_user_argument(argument)
        
        if result["success"]:
            # Update scores
            st.session_state.user_score += result["user_score"]
            st.session_state.ai_score += result["ai_score"]
            
            # Add AI response to transcript
            st.session_state.last_ai_argument = result["ai_response"]
            st.session_state.debate_transcript.append({
                'speaker': 'ai',
                'round': st.session_state.current_round,
                'content': result["ai_response"],
                'timestamp': datetime.now().isoformat(),
                'score': result["ai_score"]
            })
            
            # Update user argument with score
            for entry in st.session_state.debate_transcript:
                if entry['speaker'] == 'user' and entry['round'] == st.session_state.current_round and entry['score'] is None:
                    entry['score'] = result["user_score"]
            
            # Set judge feedback
            st.session_state.last_judge_feedback = f"User: {result['user_feedback']}\n\nAI: {result['ai_feedback']}"
            
            # Update round if complete
            if result["round_complete"]:
                st.session_state.current_round += 1
            
            # Switch back to user turn
            st.session_state.current_speaker = 'user'
            st.session_state.waiting_for_ai = False
            
            st.success("Argument submitted and processed!")
        else:
            st.error(f"Error processing argument: {result.get('error', 'Unknown error')}")
            # Fall back to simulation
            simulate_ai_response()
        
        st.rerun()
    else:
        st.error("Argument must be at least 50 characters long.")

def simulate_ai_response():
    """Simulate AI response for demo purposes."""
    # This is a placeholder - in real implementation, connect to debate system
    ai_response = "This is a simulated AI response. In the actual system, this would be generated by the debate AI."
    
    st.session_state.last_ai_argument = ai_response
    st.session_state.debate_transcript.append({
        'speaker': 'ai',
        'round': st.session_state.current_round,
        'content': ai_response,
        'timestamp': datetime.now().isoformat(),
        'score': 28
    })
    
    # Update user argument with simulated score
    for entry in st.session_state.debate_transcript:
        if entry['speaker'] == 'user' and entry['round'] == st.session_state.current_round and entry['score'] is None:
            entry['score'] = 30
    
    # Set judge feedback
    st.session_state.last_judge_feedback = "User: Good argument. AI: Standard response."
    
    # Update round and speaker
    st.session_state.current_round += 1
    st.session_state.current_speaker = 'user'
    st.session_state.waiting_for_ai = False

def record_voice_input():
    """Handle voice input recording."""
    if DEBATE_SYSTEM_AVAILABLE:
        with st.spinner("🎤 Listening..."):
            text = speech_to_text_whisper_api()
            if text:
                # Store the transcribed text in a different session state variable
                st.session_state.transcribed_text = text
                st.success(f"Transcribed: {text}")
                # Update the form value for next render
                st.session_state.user_argument_text = text
                st.rerun()
    else:
        st.warning("Voice input is not available. Please type your argument.")

def calculate_average_score(speaker: str) -> float:
    """Calculate average score for a speaker."""
    scores = [entry['score'] for entry in st.session_state.debate_transcript 
              if entry['speaker'] == speaker and entry.get('score')]
    return sum(scores) / len(scores) if scores else 0

def get_best_round(speaker: str) -> str:
    """Get the best scoring round for a speaker."""
    entries = [entry for entry in st.session_state.debate_transcript 
               if entry['speaker'] == speaker and entry.get('score')]
    if entries:
        best = max(entries, key=lambda x: x['score'])
        return f"Round {best['round']} ({best['score']}/40)"
    return "N/A"

def generate_improvement_suggestions() -> List[str]:
    """Generate improvement suggestions based on performance."""
    suggestions = []
    
    # Analyze scores
    user_avg = calculate_average_score('user')
    if user_avg > 0:
        if user_avg < 20:
            suggestions.append("Focus on strengthening logical coherence and evidence")
        elif user_avg < 30:
            suggestions.append("Good foundation - work on persuasiveness and relevance")
        else:
            suggestions.append("Excellent performance - maintain consistency")
    
    return suggestions

def save_transcript():
    """Save debate transcript to file."""
    if st.session_state.debate_transcript:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"debate_transcript_{timestamp}.json"
        
        transcript_data = {
            "topic": st.session_state.current_topic,
            "date": datetime.now().isoformat(),
            "transcript": st.session_state.debate_transcript,
            "final_scores": {
                "user": st.session_state.user_score,
                "ai": st.session_state.ai_score
            },
            "config": st.session_state.debate_config
        }
        
        json_str = json.dumps(transcript_data, indent=2)
        b64 = base64.b64encode(json_str.encode()).decode()
        
        st.sidebar.markdown(
            f'<a href="data:application/json;base64,{b64}" download="{filename}">📥 Download Transcript</a>',
            unsafe_allow_html=True
        )
        st.sidebar.success("Transcript ready for download!")
    else:
        st.sidebar.warning("No transcript to save yet.")

def export_statistics():
    """Export debate statistics."""
    stats = {
        "topic": st.session_state.current_topic,
        "rounds_completed": st.session_state.current_round - 1,
        "user_total_score": st.session_state.user_score,
        "ai_total_score": st.session_state.ai_score,
        "user_average": calculate_average_score('user'),
        "ai_average": calculate_average_score('ai'),
        "timestamp": datetime.now().isoformat()
    }
    
    json_str = json.dumps(stats, indent=2)
    b64 = base64.b64encode(json_str.encode()).decode()
    
    filename = f"debate_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    st.sidebar.markdown(
        f'<a href="data:application/json;base64,{b64}" download="{filename}">📥 Download Statistics</a>',
        unsafe_allow_html=True
    )
    st.sidebar.success("Statistics ready for download!")

# Main app
def main():
    # Load custom CSS
    load_custom_css()
    
    # Initialize session state
    init_session_state()
    
    # Update configuration with API keys
    update_config_with_api_keys()
    
    # Render header
    render_header()
    
    # Render sidebar
    render_sidebar()
    
    # Render main interface
    render_main_interface()

if __name__ == "__main__":
    main()
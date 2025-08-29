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
    page_title="Formal Debate Chamber",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional Courtroom Aesthetic
def load_custom_css():
    st.markdown("""
    <style>
    /* Import classical fonts */
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Crimson+Text:wght@400;600&display=swap');
    
    /* Main background with wood grain texture */
    .stApp {
        background: linear-gradient(
            180deg,
            rgba(40, 30, 20, 0.95) 0%,
            rgba(55, 40, 30, 0.95) 50%,
            rgba(40, 30, 20, 0.95) 100%
        );
        background-image: 
            repeating-linear-gradient(
                90deg,
                rgba(60, 45, 35, 0.1) 0px,
                rgba(80, 60, 45, 0.1) 3px,
                rgba(60, 45, 35, 0.1) 6px
            );
    }
    
    /* Header styling with brass accents */
    h1, h2, h3 {
        font-family: 'Playfair Display', serif !important;
        color: #D4AF37 !important;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
        letter-spacing: 1px;
    }
    
    /* Embossed title effect */
    .main-title {
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        color: #D4AF37;
        text-shadow: 
            0 1px 0 #ccc,
            0 2px 0 #c9c9c9,
            0 3px 0 #bbb,
            0 4px 0 #b9b9b9,
            0 5px 0 #aaa,
            0 6px 1px rgba(0,0,0,.1),
            0 0 5px rgba(0,0,0,.1),
            0 1px 3px rgba(0,0,0,.3),
            0 3px 5px rgba(0,0,0,.2),
            0 5px 10px rgba(0,0,0,.25);
        margin-bottom: 2rem;
    }
    
    /* Leather-textured buttons */
    .stButton > button {
        background: linear-gradient(145deg, #5a453a, #3e2f26);
        color: #D4AF37;
        border: 2px solid #D4AF37;
        border-radius: 8px;
        padding: 12px 24px;
        font-family: 'Crimson Text', serif;
        font-size: 1.1rem;
        font-weight: 600;
        letter-spacing: 1px;
        box-shadow: 
            inset 0 -3px 7px rgba(0,0,0,0.5),
            0 2px 3px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
        text-transform: uppercase;
    }
    
    .stButton > button:hover {
        background: linear-gradient(145deg, #6a554a, #4e3f36);
        box-shadow: 
            inset 0 -3px 7px rgba(0,0,0,0.7),
            0 4px 6px rgba(0,0,0,0.4);
        transform: translateY(-2px);
    }
    
    /* Parchment-style text areas */
    .stTextArea textarea, .stTextInput input {
        background: linear-gradient(145deg, #f5e6d3, #e8d7c3);
        color: #2a1810;
        border: 2px solid #8B7355;
        border-radius: 6px;
        font-family: 'Crimson Text', serif;
        font-size: 1.05rem;
        padding: 12px;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Score displays with brass plates */
    .score-display {
        background: linear-gradient(145deg, #D4AF37, #B8941F);
        color: #1a1510;
        padding: 15px 25px;
        border-radius: 8px;
        font-family: 'Playfair Display', serif;
        font-size: 1.8rem;
        font-weight: 700;
        text-align: center;
        box-shadow: 
            inset 0 2px 4px rgba(255,255,255,0.3),
            0 4px 8px rgba(0,0,0,0.4);
        border: 2px solid #8B7355;
    }
    
    /* Argument cards with leather texture */
    .argument-card {
        background: linear-gradient(145deg, #3a2a20, #2a1a10);
        border: 2px solid #D4AF37;
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 
            0 6px 12px rgba(0,0,0,0.3),
            inset 0 1px 0 rgba(255,255,255,0.1);
    }
    
    .argument-card h4 {
        color: #D4AF37;
        font-family: 'Playfair Display', serif;
        margin-bottom: 10px;
        font-size: 1.3rem;
    }
    
    .argument-card p {
        color: #e8d7c3;
        font-family: 'Crimson Text', serif;
        line-height: 1.6;
        font-size: 1.05rem;
    }
    
    /* Judge's gavel animation */
    @keyframes gavel-strike {
        0% { transform: rotate(0deg); }
        50% { transform: rotate(-30deg); }
        100% { transform: rotate(0deg); }
    }
    
    .gavel-icon {
        animation: gavel-strike 2s ease-in-out infinite;
        display: inline-block;
    }
    
    /* Progress bar with brass finish */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #D4AF37, #FFD700, #D4AF37);
    }
    
    /* Select boxes */
    .stSelectbox > div > div {
        background: linear-gradient(145deg, #3a2a20, #2a1a10);
        border: 2px solid #D4AF37;
        color: #D4AF37;
    }
    
    /* Metrics styling */
    [data-testid="metric-container"] {
        background: linear-gradient(145deg, #2a2015, #1a1510);
        border: 2px solid #D4AF37;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(145deg, #3a2a20, #2a1a10);
        border: 2px solid #D4AF37;
        border-radius: 8px;
        color: #D4AF37 !important;
        font-family: 'Crimson Text', serif;
    }
    
    /* Loading spinner */
    .stSpinner > div {
        border-color: #D4AF37 !important;
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
        st.session_state.audio_enabled = MURF_CONFIG["api_key"] is not None if DEBATE_SYSTEM_AVAILABLE else False
    if 'voice_input_enabled' not in st.session_state:
        st.session_state.voice_input_enabled = CONFIG["openai_api_key"] is not None if DEBATE_SYSTEM_AVAILABLE else False
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

# Header with gavel icon
def render_header():
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.markdown("""
        <div class="main-title">
            ⚖️ FORMAL DEBATE CHAMBER ⚖️
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<center><i>A Professional Forum for Intellectual Discourse</i></center>", unsafe_allow_html=True)

# Sidebar configuration
def render_sidebar():
    with st.sidebar:
        st.markdown("## ⚙️ Chamber Configuration")
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
            "Select a topic:",
            preset_topics,
            key="topic_preset"
        )
        
        if selected_preset == "Custom Topic":
            topic = st.text_area(
                "Enter your custom topic:",
                height=100,
                key="custom_topic"
            )
        else:
            topic = selected_preset
            
        st.markdown("---")
        
        # Debate settings
        st.markdown("### ⚡ Debate Settings")
        
        # Use separate variables instead of directly modifying session state
        max_rounds = st.slider(
            "Number of Rounds:",
            min_value=1,
            max_value=5,
            value=st.session_state.debate_config["max_rounds"],
            key="max_rounds_slider"
        )
        
        time_limit = st.slider(
            "Time per Argument (seconds):",
            min_value=30,
            max_value=300,
            value=st.session_state.debate_config["time_limit"],
            step=30,
            key="time_limit_slider"
        )
        
        # Update config without conflicting with widget keys
        st.session_state.debate_config["max_rounds"] = max_rounds
        st.session_state.debate_config["time_limit"] = time_limit
        
        st.markdown("---")
        
        # Audio settings
        st.markdown("### 🔊 Audio Configuration")
        
        audio_enabled = st.checkbox(
            "Enable Voice Output",
            value=st.session_state.audio_enabled,
            disabled=not (DEBATE_SYSTEM_AVAILABLE and MURF_CONFIG.get("api_key")) if DEBATE_SYSTEM_AVAILABLE else True
        )
        st.session_state.audio_enabled = audio_enabled
        
        voice_input = st.checkbox(
            "Enable Voice Input",
            value=st.session_state.voice_input_enabled,
            disabled=not (DEBATE_SYSTEM_AVAILABLE and CONFIG.get("openai_api_key")) if DEBATE_SYSTEM_AVAILABLE else True
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
        
        # Start debate button
        if not st.session_state.debate_active:
            if st.button("🔨 **COMMENCE DEBATE**", use_container_width=True):
                if topic and topic != "Custom Topic":
                    start_debate(topic, max_rounds)
                else:
                    st.error("Please select or enter a debate topic.")
        else:
            if st.button("⛔ **ADJOURN DEBATE**", use_container_width=True):
                stop_debate()
        
        # Export options
        st.markdown("---")
        st.markdown("### 📁 Export Options")
        
        if st.button("💾 Save Transcript", use_container_width=True):
            save_transcript()
        
        if st.button("📊 Export Statistics", use_container_width=True):
            export_statistics()

# Main debate interface
def render_main_interface():
    if not st.session_state.debate_active:
        render_welcome_screen()
    else:
        render_debate_screen()

# Welcome screen
def render_welcome_screen():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 40px;'>
            <h2>Welcome to the Formal Debate Chamber</h2>
            <p style='font-size: 1.2rem; color: #D4AF37; font-family: Crimson Text, serif;'>
                A distinguished platform for structured intellectual discourse,
                featuring AI-powered opponents and impartial adjudication.
            </p>
            <br>
            <p style='font-size: 1.1rem; color: #e8d7c3;'>
                Configure your debate parameters in the sidebar and click
                <b>COMMENCE DEBATE</b> to begin.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature highlights
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class='argument-card'>
                <h4>🎙️ Voice Integration</h4>
                <p>Speak your arguments naturally with voice input and hear AI responses.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class='argument-card'>
                <h4>⚖️ Impartial Judging</h4>
                <p>AI-powered evaluation based on logic, evidence, and persuasiveness.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class='argument-card'>
                <h4>📊 Detailed Analytics</h4>
                <p>Comprehensive scoring and feedback for continuous improvement.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # System status
        if not DEBATE_SYSTEM_AVAILABLE:
            st.warning("⚠️ Debate system not fully loaded. Some features may be unavailable.")
        if not DEBATE_INTEGRATION_AVAILABLE:
            st.warning("⚠️ Debate integration not available. Using simulation mode.")

# Active debate screen
def render_debate_screen():
    # Debate header info
    st.markdown(f"""
    <div style='background: linear-gradient(145deg, #2a2015, #1a1510); 
                border: 3px solid #D4AF37; 
                border-radius: 10px; 
                padding: 20px; 
                margin-bottom: 20px;'>
        <h3 style='text-align: center; margin: 0;'>Current Debate</h3>
        <p style='text-align: center; font-size: 1.2rem; color: #e8d7c3; margin-top: 10px;'>
            {st.session_state.current_topic}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Progress and scores
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        st.markdown(f"""
        <div class='score-display'>
            <div>USER</div>
            <div style='font-size: 2.5rem;'>{st.session_state.user_score}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        current_round = st.session_state.current_round
        max_rounds = st.session_state.debate_config["max_rounds"]
        progress = min(current_round / max_rounds, 1.0)
        st.progress(progress)
        st.markdown(f"<center><b>Round {current_round} of {max_rounds}</b></center>", unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class='score-display'>
            <div>AI</div>
            <div style='font-size: 2.5rem;'>{st.session_state.ai_score}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Debate content area
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
        st.info(st.session_state.user_position or "Position will be assigned when debate starts")
        
        st.markdown("### 💬 Your Argument")
        
        # Check if it's user's turn
        if st.session_state.current_speaker == 'user' and not st.session_state.waiting_for_ai:
            # Create a form for argument submission
            with st.form(key="argument_form"):
                user_argument = st.text_area(
                    "Type your argument:",
                    height=150,
                    key="user_argument_text",
                    value=st.session_state.get('transcribed_text', st.session_state.get('user_argument_text', ''))
                )
                
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    submit_button = st.form_submit_button("**SUBMIT ARGUMENT**", use_container_width=True)
                with col_b:
                    if st.session_state.voice_input_enabled:
                        voice_button = st.form_submit_button("🎤 Record", use_container_width=True)
                    else:
                        voice_button = False
                
                if submit_button and user_argument:
                    submit_user_argument(user_argument)
                    # Clear transcribed text after submission
                    if 'transcribed_text' in st.session_state:
                        st.session_state.transcribed_text = ""
                elif voice_button:
                    record_voice_input()
        else:
            if st.session_state.waiting_for_ai:
                st.info("⏳ Waiting for AI response...")
            else:
                st.info("⏳ Waiting for your turn...")
    
    with col2:
        st.markdown("### 🤖 AI Position")
        st.info(st.session_state.ai_position or "Position will be assigned when debate starts")
        
        st.markdown("### 💭 AI Argument")
        
        if st.session_state.last_ai_argument:
            st.markdown(f"""
            <div class='argument-card'>
                <p>{st.session_state.last_ai_argument}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("⏳ AI preparing argument...")
    
    # Judge feedback section
    st.markdown("---")
    st.markdown("### ⚖️ Judge's Evaluation")
    
    if st.session_state.last_judge_feedback:
        st.success(st.session_state.last_judge_feedback)
    else:
        st.info("Awaiting arguments for evaluation...")

# Transcript tab
def render_transcript():
    if st.session_state.debate_transcript:
        for entry in st.session_state.debate_transcript:
            if entry['speaker'] == 'user':
                st.markdown(f"""
                <div class='argument-card' style='border-color: #4CAF50;'>
                    <h4>👤 User - Round {entry['round']}</h4>
                    <p>{entry['content']}</p>
                    <small>Score: {entry.get('score', 'Pending')}/40</small>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='argument-card' style='border-color: #2196F3;'>
                    <h4>🤖 AI - Round {entry['round']}</h4>
                    <p>{entry['content']}</p>
                    <small>Score: {entry.get('score', 'Pending')}/40</small>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No arguments yet. The transcript will appear here as the debate progresses.")

# Analysis tab
def render_analysis():
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Score Progression")
        if len(st.session_state.debate_transcript) > 0:
            # Simple score visualization
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
                st.bar_chart({"User": user_scores, "AI": ai_scores})
        else:
            st.info("Score analysis will appear after the first scored round.")
    
    with col2:
        st.markdown("### 🎯 Key Metrics")
        col_a, col_b = st.columns(2)
        
        with col_a:
            avg_user = calculate_average_score('user')
            st.metric("Avg. User Score", f"{avg_user:.1f}")
            st.metric("Best User Round", get_best_round('user'))
        
        with col_b:
            avg_ai = calculate_average_score('ai')
            st.metric("Avg. AI Score", f"{avg_ai:.1f}")
            st.metric("Best AI Round", get_best_round('ai'))
    
    st.markdown("---")
    st.markdown("### 💡 Improvement Suggestions")
    
    suggestions = generate_improvement_suggestions()
    if suggestions:
        for suggestion in suggestions:
            st.warning(f"• {suggestion}")
    else:
        st.info("Suggestions will appear based on judge feedback.")

# Helper functions
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
    
    # Render header
    render_header()
    
    # Render sidebar
    render_sidebar()
    
    # Render main interface
    render_main_interface()

if __name__ == "__main__":
    main()
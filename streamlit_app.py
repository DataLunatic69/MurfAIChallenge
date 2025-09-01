# streamlit_app.py - Enhanced Dynamic UI
import streamlit as st
import time
import json
from datetime import datetime
import threading
import queue
import asyncio
from typing import Optional, Dict, Any
import speech_recognition as sr
import tempfile
import os

# Import from run.py
from run import (
    CONFIG, MURF_CONFIG, WHISPER_CONFIG,
    get_llm, llm, judge_llm,
    SpeakerType, DebatePhase, Argument, Score, DebateContext,
    speak_text_streaming, audio_streamer,
    speech_to_text_whisper_api, openai_client,
    extract_key_points, detect_repetition, analyze_opponent_argument,
    JudgeScore, ArgumentAnalysis
)

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser

# Page configuration
st.set_page_config(
    page_title="Murf Debate Chamber",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
def inject_custom_css():
    st.markdown("""
    <style>
    .stApp { 
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: #ffffff;
    }
    
    .debate-header {
        background: linear-gradient(135deg, #0f3460, #16213e);
        color: #ffffff;
        padding: 2.5rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .debate-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: #ffffff;
    }
    
    .debate-header p {
        font-size: 1.1rem;
        opacity: 0.9;
        color: #ffffff;
    }
    
    .user-arg {
        background: rgba(25, 118, 210, 0.1);
        border: 2px solid #1976d2;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        color: #ffffff;
        box-shadow: 0 4px 15px rgba(25, 118, 210, 0.2);
    }
    
    .user-arg h4 {
        color: #64b5f6;
        margin-bottom: 1rem;
    }
    
    .user-arg p {
        color: #ffffff;
        line-height: 1.6;
    }
    
    .ai-arg {
        background: rgba(105, 159, 56, 0.1);
        border: 2px solid #689f38;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        color: #ffffff;
        box-shadow: 0 4px 15px rgba(105, 159, 56, 0.2);
    }
    
    .ai-arg h4 {
        color: #8bc34a;
        margin-bottom: 1rem;
    }
    
    .ai-arg p {
        color: #ffffff;
        line-height: 1.6;
    }
    
    .judge-feedback {
        background: rgba(255, 152, 0, 0.1);
        border: 2px solid #ff9800;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        color: #ffffff;
        box-shadow: 0 4px 15px rgba(255, 152, 0, 0.2);
    }
    
    .judge-feedback h4 {
        color: #ffb74d;
        margin-bottom: 1rem;
    }
    
    .judge-feedback p {
        color: #ffffff;
        line-height: 1.6;
    }
    
    .score-display {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        backdrop-filter: blur(10px);
        color: #ffffff;
    }
    
    .score-display h3 {
        color: #64b5f6;
        margin-bottom: 1rem;
    }
    
    .score-display h1 {
        color: #ffffff;
        font-size: 3rem;
        font-weight: 700;
    }
    
    .speaking-indicator {
        display: inline-block;
        padding: 0.5rem 1rem;
        background: #4caf50;
        color: white;
        border-radius: 20px;
        animation: pulse 1.5s infinite;
        font-weight: bold;
    }
    
    .listening-indicator {
        display: inline-block;
        padding: 0.5rem 1rem;
        background: #ff9800;
        color: white;
        border-radius: 20px;
        animation: pulse 1.5s infinite;
        font-weight: bold;
    }
    
    .round-indicator {
        background: linear-gradient(135deg, #3498db, #2980b9);
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        margin: 0.5rem;
        display: inline-block;
        font-weight: 600;
        box-shadow: 0 3px 10px rgba(52, 152, 219, 0.3);
    }
    
    .status-active { 
        color: #4caf50; 
        font-weight: bold; 
        background: rgba(76, 175, 80, 0.1);
        padding: 0.5rem;
        border-radius: 5px;
    }
    
    .status-waiting { 
        color: #ff9800; 
        font-weight: bold; 
        background: rgba(255, 152, 0, 0.1);
        padding: 0.5rem;
        border-radius: 5px;
    }
    
    .status-complete { 
        color: #2196f3; 
        font-weight: bold; 
        background: rgba(33, 150, 243, 0.1);
        padding: 0.5rem;
        border-radius: 5px;
    }
    
    .turn-alert {
        background: linear-gradient(135deg, #00b09b, #96c93d);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,176,155,0.3);
        animation: pulseGlow 2s infinite;
    }
    
    @keyframes pulseGlow {
        0%, 100% { box-shadow: 0 10px 30px rgba(0,176,155,0.3); }
        50% { box-shadow: 0 10px 40px rgba(0,176,155,0.5); }
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.6; }
        100% { opacity: 1; }
    }
    
    @keyframes slideIn {
        from { 
            opacity: 0;
            transform: translateX(-20px);
        }
        to { 
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    .voice-input-area {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
        backdrop-filter: blur(10px);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        border-radius: 8px;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(102,126,234,0.4);
    }
    
    .stTextArea textarea {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.2);
        color: white;
        font-size: 1.05rem;
        line-height: 1.6;
    }
    
    .stSelectbox > div > div {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.2);
        color: white;
    }
    
    .stCheckbox > div > div {
        color: #ffffff;
    }
    
    .stSlider > div > div {
        color: #ffffff;
    }
    
    .stMetric > div > div {
        color: #ffffff;
    }
    
    .stInfo {
        background: rgba(33, 150, 243, 0.1);
        border: 1px solid rgba(33, 150, 243, 0.3);
        color: #64b5f6;
    }
    
    .stSuccess {
        background: rgba(76, 175, 80, 0.1);
        border: 1px solid rgba(76, 175, 80, 0.3);
        color: #81c784;
    }
    
    .stError {
        background: rgba(244, 67, 54, 0.1);
        border: 1px solid rgba(244, 67, 54, 0.3);
        color: #e57373;
    }
    
    .stWarning {
        background: rgba(255, 152, 0, 0.1);
        border: 1px solid rgba(255, 152, 0, 0.3);
        color: #ffb74d;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    defaults = {
        'debate_active': False,
        'debate_phase': 'setup',
        'topic': '',
        'user_position': '',
        'ai_position': '',
        'arguments': [],
        'scores': [],
        'current_round': 1,
        'max_rounds': 3,
        'current_speaker': None,
        'user_input_ready': False,
        'user_argument': '',
        'ai_argument': '',
        'judge_feedback': '',
        'waiting_for_user': False,
        'debate_context': None,
        'first_speaker': None,
        'turn_count': 0,
        'final_winner': None,
        'is_listening': False,
        'voice_enabled': True,
        'auto_advance': True,
        'turn_announced': False,
        'last_user_evaluation': None,
        'last_ai_evaluation': None
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Voice input handler
def capture_voice_input():
    """Capture voice input using Whisper API"""
    if not openai_client:
        st.error("OpenAI API key not configured for voice input")
        return None
    
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    
    try:
        with microphone as source:
            st.session_state.is_listening = True
            recognizer.adjust_for_ambient_noise(source, duration=1.0)
            
            # Create placeholder for listening status
            with st.spinner("🎤 Listening... Speak now!"):
                audio = recognizer.listen(
                    source, 
                    timeout=WHISPER_CONFIG["timeout"],
                    phrase_time_limit=WHISPER_CONFIG["phrase_time_limit"]
                )
        
        st.session_state.is_listening = False
        
        # Save audio to temp file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            with open(tmp.name, 'wb') as f:
                f.write(audio.get_wav_data())
            audio_path = tmp.name
        
        # Transcribe with Whisper
        with st.spinner("Processing speech..."):
            with open(audio_path, 'rb') as audio_file:
                transcript = openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="en",
                    response_format="text"
                )
        
        os.unlink(audio_path)
        
        if transcript and transcript.strip():
            return transcript.strip()
        return None
        
    except sr.WaitTimeoutError:
        st.warning("No speech detected. Please try again.")
        return None
    except Exception as e:
        st.error(f"Speech recognition error: {e}")
        return None
    finally:
        st.session_state.is_listening = False

# Main debate controller
class DebateController:
    def __init__(self):
        self.llm = get_llm()
        self.judge_llm = get_llm()
    
    def setup_debate(self, topic):
        """Initialize debate with topic and positions"""
        st.session_state.topic = topic
        
        # Generate positions
        prompt = f"""Given the debate topic: "{topic}"
        What are the two main opposing positions?
        Format: 
        FOR: [position supporting the proposition]
        AGAINST: [position opposing the proposition]"""
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        lines = response.content.strip().split('\n')
        
        st.session_state.user_position = lines[0].replace("FOR:", "").strip() if lines else "In favor"
        st.session_state.ai_position = lines[1].replace("AGAINST:", "").strip() if len(lines) > 1 else "Against"
        
        # Initialize debate context
        st.session_state.debate_context = {
            'user_main_points': [],
            'ai_main_points': [],
            'unaddressed_points': {'user': [], 'ai': []},
            'evidence_used': {'user': [], 'ai': []}
        }
        
        # Select first speaker
        import random
        st.session_state.first_speaker = random.choice(['user', 'ai'])
        st.session_state.current_speaker = st.session_state.first_speaker
        st.session_state.debate_phase = 'debate'
        st.session_state.debate_active = True
        
        return True
    
    def process_user_argument(self, argument_text):
        """Process and validate user argument"""
        if len(argument_text) < 20:
            return False, "Argument too short. Please provide more detail (at least 20 characters)."
        
        # Check for repetition
        is_repetitive, similar_to = detect_repetition(
            argument_text, 
            st.session_state.arguments, 
            'user'
        )
        
        if is_repetitive:
            st.warning(f"This seems similar to a previous argument. Consider a new angle.")
        
        # Extract key points
        key_points = extract_key_points(argument_text, self.llm)
        
        # Create argument record
        argument = {
            'speaker': 'user',
            'content': argument_text,
            'timestamp': time.time(),
            'round_number': st.session_state.current_round,
            'key_points': key_points,
            'rebuts_points': []
        }
        
        st.session_state.arguments.append(argument)
        st.session_state.debate_context['user_main_points'].extend(key_points)
        st.session_state.turn_count += 1
        
        return True, "Argument processed successfully"
    
    def generate_ai_argument(self):
        """Generate AI's debate argument"""
        recent_user_args = [
            arg for arg in st.session_state.arguments 
            if arg['speaker'] == 'user'
        ]
        
        opponent_analysis = None
        if recent_user_args:
            last_user_arg = recent_user_args[-1]
            opponent_analysis = analyze_opponent_argument(
                last_user_arg['content'], 
                self.llm
            )
        
        # Build prompt
        system_prompt = f"""You are participating in a formal debate.
Topic: {st.session_state.topic}
Your position: {st.session_state.ai_position}
Current round: {st.session_state.current_round} of {st.session_state.max_rounds}

IMPORTANT: 
- Keep your argument CONCISE and to the point (2-3 sentences maximum)
- Make NEW arguments, address opponent's claims, use different evidence
- Be direct and avoid unnecessary elaboration
- Focus on the strongest points only"""
        
        user_prompt = ""
        if opponent_analysis:
            user_prompt += f"""Opponent's main claims:
{chr(10).join([f"- {claim}" for claim in opponent_analysis.main_claims])}

Weak points to address:
{chr(10).join([f"- {weak}" for weak in opponent_analysis.weak_points])}

"""
        
        round_type = "Opening argument" if st.session_state.current_round == 1 else \
                    "Rebuttal" if st.session_state.current_round == 2 else "Closing"
        user_prompt += f"Provide your {round_type}:"
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = self.llm.invoke(messages)
        ai_argument = response.content
        
        # Extract key points
        key_points = extract_key_points(ai_argument, self.llm)
        
        # Create argument record
        argument = {
            'speaker': 'ai',
            'content': ai_argument,
            'timestamp': time.time(),
            'round_number': st.session_state.current_round,
            'key_points': key_points,
            'rebuts_points': opponent_analysis.main_claims[:2] if opponent_analysis else []
        }
        
        st.session_state.arguments.append(argument)
        st.session_state.debate_context['ai_main_points'].extend(key_points)
        st.session_state.turn_count += 1
        st.session_state.ai_argument = ai_argument
        
        return ai_argument
    
    def judge_argument(self, argument, speaker):
        """Judge the most recent argument"""
        parser = PydanticOutputParser(pydantic_object=JudgeScore)
        
        system_prompt = f"""You are an impartial debate judge. Score each criterion from 1-10.

Scoring criteria:
- Logical coherence: Clear reasoning and structure
- Evidence/support: Use of facts, examples, data
- Relevance: Directly addresses the topic
- Persuasiveness: Compelling and convincing delivery

{parser.get_format_instructions()}"""
        
        position = st.session_state.user_position if speaker == 'user' else st.session_state.ai_position
        
        user_prompt = f"""Topic: {st.session_state.topic}
Speaker: {speaker}
Position: {position}
Round: {st.session_state.current_round}

Argument to evaluate:
{argument}

Provide your evaluation:"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        try:
            response = self.judge_llm.invoke(messages)
            evaluation = parser.parse(response.content)
        except:
            evaluation = JudgeScore(
                logical_coherence=7,
                evidence_support=6,
                relevance=8,
                persuasiveness=7,
                total_score=28,
                reasoning="Standard evaluation applied.",
                strengths=["Clear argument"],
                weaknesses=["Could use more evidence"]
            )
        
        score = {
            'user_score': evaluation.total_score if speaker == 'user' else 0,
            'ai_score': evaluation.total_score if speaker == 'ai' else 0,
            'round_number': st.session_state.current_round,
            'reasoning': evaluation.reasoning,
            'detailed_scores': {
                'logical_coherence': evaluation.logical_coherence,
                'evidence_support': evaluation.evidence_support,
                'relevance': evaluation.relevance,
                'persuasiveness': evaluation.persuasiveness
            }
        }
        
        st.session_state.scores.append(score)
        st.session_state.judge_feedback = evaluation.reasoning
        
        return evaluation
    
    def check_round_complete(self):
        """Check if current round is complete"""
        speakers_in_round = [
            arg['speaker'] for arg in st.session_state.arguments 
            if arg['round_number'] == st.session_state.current_round
        ]
        
        is_complete = 'user' in speakers_in_round and 'ai' in speakers_in_round
        
        # Debug logging
        print(f"Round {st.session_state.current_round} check:")
        print(f"  Arguments in this round: {speakers_in_round}")
        print(f"  Is complete: {is_complete}")
        
        return is_complete
    
    def advance_round(self):
        """Move to next round or end debate"""
        print(f"Advancing round from {st.session_state.current_round} to {st.session_state.current_round + 1}")
        
        if st.session_state.current_round >= st.session_state.max_rounds:
            print("Max rounds reached, ending debate")
            self.end_debate()
        else:
            st.session_state.current_round += 1
            # Alternate who goes first each round - if AI went first in round 1, user goes first in round 2, etc.
            if st.session_state.current_round % 2 == 1:  # Odd rounds (1, 3, 5...)
                st.session_state.current_speaker = st.session_state.first_speaker
            else:  # Even rounds (2, 4, 6...)
                st.session_state.current_speaker = 'user' if st.session_state.first_speaker == 'ai' else 'ai'
            print(f"Advanced to round {st.session_state.current_round}, next speaker: {st.session_state.current_speaker}")
    
    def end_debate(self):
        """Calculate final scores and end debate"""
        total_user = sum(s['user_score'] for s in st.session_state.scores)
        total_ai = sum(s['ai_score'] for s in st.session_state.scores)
        
        if total_user > total_ai:
            st.session_state.final_winner = f"User wins by {total_user - total_ai} points!"
        elif total_ai > total_user:
            st.session_state.final_winner = f"AI wins by {total_ai - total_user} points!"
        else:
            st.session_state.final_winner = "It's a tie!"
        
        st.session_state.debate_active = False
        st.session_state.debate_phase = 'complete'

# UI Components
def render_header():
            st.markdown("""
    <div class="debate-header">
        <h1>âš–ï¸ AI Debate Chamber</h1>
        <p>Real-time formal debate platform with voice interaction</p>
            </div>
            """, unsafe_allow_html=True)
    
def render_sidebar():
    with st.sidebar:
        st.markdown("### âš™ï¸ Debate Configuration")
        
        topics = [
            "Should AI be regulated by the government?",
            "Is remote work better than office work?",
            "Should social media be held responsible for content?",
            "Is UBI necessary for the future?",
            "Custom topic"
        ]
        
        selected = st.selectbox("Select Topic", topics)
        
        if selected == "Custom topic":
            custom = st.text_input("Enter your topic:")
            topic = custom if custom else topics[0]
        else:
            topic = selected
        
        st.session_state.max_rounds = st.slider("Number of Rounds", 1, 5, 3)
        
        st.markdown("---")
        st.markdown("### ðŸŽ™ï¸ Voice Settings")
        
        st.session_state.voice_enabled = st.checkbox("Enable Voice Input", value=True)
        st.session_state.auto_advance = st.checkbox("Auto-advance Rounds", value=True)
        
        st.markdown("---")
        st.markdown("### ðŸ“Š Status")
        
        col1, col2 = st.columns(2)
        with col1:
            if CONFIG.get("openai_api_key"):
                st.success("OpenAI âœ“")
            else:
                st.error("OpenAI âœ—")
        
        with col2:
            if MURF_CONFIG.get("api_key"):
                st.success("Murf AI âœ“")
            else:

                st.error("Murf AI âœ—")
        
        return topic

def render_debate_status():
    """Show current debate status"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"<div class='round-indicator'>Round {st.session_state.current_round}/{st.session_state.max_rounds}</div>", 
                   unsafe_allow_html=True)
    
    with col2:
        phase_text = st.session_state.debate_phase.title()
        st.markdown(f"<div class='status-active'>Phase: {phase_text}</div>", unsafe_allow_html=True)
    
    with col3:
        if st.session_state.current_speaker:
            speaker = "You" if st.session_state.current_speaker == 'user' else "AI"
            # Add prominent turn indicator
            if st.session_state.current_speaker == 'user':
                st.markdown(f"""
                <div style='background: #e8f5e8; border: 2px solid #28a745; border-radius: 10px; padding: 10px; text-align: center;'>
                    <h4 style='color: #155724; margin: 0;'>🎯 YOUR TURN!</h4>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style='background: #fff3cd; border: 2px solid #ffc107; border-radius: 10px; padding: 10px; text-align: center;'>
                    <h4 style='color: #856404; margin: 0;'>🤖 AI's Turn</h4>
                </div>
                """, unsafe_allow_html=True)
            st.markdown(f"<div class='status-waiting'>Turn: {speaker}</div>", unsafe_allow_html=True)
    
    with col4:
        total_args = len(st.session_state.arguments)
        st.markdown(f"<div class='status-complete'>Arguments: {total_args}</div>", unsafe_allow_html=True)
    
    # Debug information
    if st.session_state.debate_active:
        st.markdown("---")
        st.markdown("### ðŸ” Debug Info")
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Current Speaker:** {st.session_state.current_speaker}")
            st.info(f"**Round:** {st.session_state.current_round}")
            st.info(f"**Turn Count:** {st.session_state.turn_count}")
        with col2:
            user_args = [arg for arg in st.session_state.arguments if arg['speaker'] == 'user' and arg['round_number'] == st.session_state.current_round]
            ai_args = [arg for arg in st.session_state.arguments if arg['speaker'] == 'ai' and arg['round_number'] == st.session_state.current_round]
            st.info(f"**User args this round:** {len(user_args)}")
            st.info(f"**AI args this round:** {len(ai_args)}")
            st.info(f"**Round complete:** {len(user_args) > 0 and len(ai_args) > 0}")

def render_score_display():
    """Display current scores"""
    col1, col2 = st.columns(2)
    
    total_user = sum(s['user_score'] for s in st.session_state.scores)
    total_ai = sum(s['ai_score'] for s in st.session_state.scores)
    
    with col1:
            st.markdown(f"""
        <div class='score-display'>
            <h3>ðŸ‘¤ Your Score</h3>
            <h1>{total_user}</h1>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='score-display'>
                         <h3>🤖 AI Score</h3>
            <h1>{total_ai}</h1>
        </div>
        """, unsafe_allow_html=True)

def render_user_turn_reminder():
    """Announce when it's the user's turn with voice"""
    if (st.session_state.current_speaker == 'user' and 
        st.session_state.debate_active and 
        MURF_CONFIG.get("api_key") and
        'turn_announced' not in st.session_state):
        
        # Announce user's turn
        reminder_text = f"It's your turn to speak in round {st.session_state.current_round}. Present your argument."
        speak_text_streaming(reminder_text)
        audio_streamer.wait_until_complete()
        st.session_state.turn_announced = True

def render_user_input_section(controller):
    """Render user input area"""
    if st.session_state.current_speaker == 'user' and st.session_state.debate_active:
        # Clear turn indicator - prominent display when it's user's turn
        st.markdown("""
        <div style='background: #e8f5e8; border: 3px solid #28a745; border-radius: 15px; padding: 20px; margin: 20px 0; text-align: center;'>
                         <h2 style='color: #155724; margin: 0;'>🎯 YOUR TURN TO SPEAK!</h2>
            <p style='color: #155724; font-size: 18px; margin: 10px 0;'>Present your argument for: <strong>{}</strong></p>
            <p style='color: #155724; font-size: 16px; margin: 5px 0;'>Round {}</p>
        </div>
        """.format(st.session_state.user_position, st.session_state.current_round), unsafe_allow_html=True)
        
        # Stop any audio playback
        audio_streamer.stop()
    
        col1, col2 = st.columns([1, 3])
        
        with col1:
            if st.session_state.voice_enabled and openai_client:
                if st.button("🎤 Voice Input", use_container_width=True, key="voice_btn"):
                    with st.spinner("Listening..."):
                        text = capture_voice_input()
                        if text:
                            st.session_state.user_argument = text
                            st.rerun()

        with col2:
            text_input = st.text_area(
                "Or type your argument:",
                value=st.session_state.user_argument,
                height=100,
                key="text_input"
            )
            
            if text_input:
                st.session_state.user_argument = text_input
        
        if st.session_state.user_argument:
            st.info(f"Your argument: {st.session_state.user_argument[:200]}...")
            
            if st.button("Submit Argument", type="primary", use_container_width=True):
                success, msg = controller.process_user_argument(st.session_state.user_argument)
                if success:
                    # Judge the argument
                    with st.spinner("Judge is evaluating..."):
                        evaluation = controller.judge_argument(st.session_state.user_argument, 'user')
                        
                        # Store judge evaluation for transcript
                        st.session_state.last_user_evaluation = {
                            'reasoning': evaluation.reasoning,
                            'score': evaluation.total_score,
                            'round': st.session_state.current_round,
                            'timestamp': time.time()
                        }
                        
                        # Speak judge feedback
                        if MURF_CONFIG["api_key"]:
                            speak_text_streaming(f"Judge feedback: {evaluation.reasoning}")
                            audio_streamer.wait_until_complete()
                    
                    # Clear input and switch speakers
                    st.session_state.user_argument = ""
                    
                    # Switch to AI's turn
                    st.session_state.current_speaker = 'ai'
                    st.session_state.turn_announced = False  # Reset for next AI turn
                    
                    # Check if round is complete after both speakers
                    if controller.check_round_complete():
                        controller.advance_round()
                    
                    st.rerun()
                else:
                    st.error(msg)

def render_ai_turn(controller):
    """Handle AI's turn"""
    if st.session_state.current_speaker == 'ai' and st.session_state.debate_active:
        with st.spinner("AI is formulating response..."):
            time.sleep(2)  # Dramatic pause
            
            # Generate AI argument
            ai_argument = controller.generate_ai_argument()
            
            # Display AI argument
            st.markdown(f"""
            <div class='ai-arg'>
                                 <h4>🤖 AI Debater (Round {st.session_state.current_round})</h4>
                <p>{ai_argument}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Speak AI argument
            if MURF_CONFIG["api_key"]:
                with st.spinner("AI is speaking..."):
                    speak_text_streaming(ai_argument)
                    audio_streamer.wait_until_complete()
            
            # Judge the argument
            with st.spinner("Judge is evaluating..."):
                time.sleep(1)
                evaluation = controller.judge_argument(ai_argument, 'ai')
                
                # Store judge evaluation for transcript
                st.session_state.last_ai_evaluation = {
                    'reasoning': evaluation.reasoning,
                    'score': evaluation.total_score,
                    'round': st.session_state.current_round,
                    'timestamp': time.time()
                }
                
                # Display and speak judge feedback
                st.markdown(f"""
                <div class='judge-feedback'>
                    <h4>âš–ï¸ Judge Feedback</h4>
                    <p>{evaluation.reasoning}</p>
                    <p><strong>Score:</strong> {evaluation.total_score}/40</p>
                </div>
                """, unsafe_allow_html=True)
                
                if MURF_CONFIG["api_key"]:
                    speak_text_streaming(f"Judge feedback: {evaluation.reasoning}")
                    audio_streamer.wait_until_complete()
            
            # Switch back to user's turn
            st.session_state.current_speaker = 'user'
            st.session_state.turn_announced = False  # Reset for next user turn
            
            # Check if round is complete after both speakers
            if controller.check_round_complete():
                controller.advance_round()
            
            # Auto-advance if enabled
            if st.session_state.auto_advance:
                time.sleep(2)
                st.rerun()

def render_stored_judge_evaluations():
    """Display stored judge evaluations from current session"""
    if st.session_state.get('last_user_evaluation'):
        st.markdown("---")
        st.markdown("### ⚖️ Judge's Evaluation - Your Argument")
        st.markdown(f"""
        <div class='judge-feedback'>
            <h4>⚖️ Judge Feedback</h4>
            <p>{st.session_state.last_user_evaluation['reasoning']}</p>
            <p><strong>Score:</strong> {st.session_state.last_user_evaluation['score']}/40</p>
        </div>
        """, unsafe_allow_html=True)
    
    if st.session_state.get('last_ai_evaluation'):
        st.markdown("---")
        st.markdown("### ⚖️ Judge's Evaluation - AI Argument")
        st.markdown(f"""
        <div class='judge-feedback'>
            <h4>⚖️ Judge Feedback</h4>
            <p>{st.session_state.last_ai_evaluation['reasoning']}</p>
            <p><strong>Score:</strong> {st.session_state.last_ai_evaluation['score']}/40</p>
        </div>
        """, unsafe_allow_html=True)

def render_debate_transcript():
    """Display debate transcript"""
    if st.session_state.arguments:
        st.markdown("### ðŸ“œ Debate Transcript")
        
        for arg in st.session_state.arguments:
            if arg['speaker'] == 'user':
                st.markdown(f"""
                <div class='user-arg'>
                    <h4>ðŸ‘¤ You (Round {arg['round_number']})</h4>
                    <p>{arg['content']}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
                st.markdown(f"""
                <div class='ai-arg'>
                    <h4>🤖 AI (Round {arg['round_number']})</h4>
                    <p>{arg['content']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Show judge feedback for this argument
                relevant_score = next(
                (s for s in st.session_state.scores 
                 if s['round_number'] == arg['round_number'] and 
                 (s['user_score'] > 0 if arg['speaker'] == 'user' else s['ai_score'] > 0)),
                None
            )
            
        if relevant_score:
                score = relevant_score['user_score'] if arg['speaker'] == 'user' else relevant_score['ai_score']
                st.markdown(f"""
                <div class='judge-feedback'>
                    <strong>Judge:</strong> {relevant_score['reasoning']}<br>
                    <strong>Score:</strong> {score}/40
                </div>
                """, unsafe_allow_html=True)

def render_final_results():
    """Display final debate results"""
    if st.session_state.debate_phase == 'complete':
        st.markdown("### ðŸ† Final Results")
        
        total_user = sum(s['user_score'] for s in st.session_state.scores)
        total_ai = sum(s['ai_score'] for s in st.session_state.scores)
        
        st.markdown(f"""
        <div class='debate-header'>
            <h2>{st.session_state.final_winner}</h2>
            <p>Final Scores - You: {total_user} | AI: {total_ai}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Announce winner with voice
        if MURF_CONFIG["api_key"]:
            speak_text_streaming(f"The debate has concluded. {st.session_state.final_winner}")
            audio_streamer.wait_until_complete()

# Main Application
def main():
    inject_custom_css()
    init_session_state()
    render_header()
    
    topic = render_sidebar()
    controller = DebateController()
    
    # Main control area
    if not st.session_state.debate_active and st.session_state.debate_phase != 'complete':
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f"### Selected Topic: {topic}")
            if st.button("🚀 Start Debate", type="primary", use_container_width=True):
                with st.spinner("Setting up debate..."):
                    controller.setup_debate(topic)
                st.rerun()
    
    # Active debate area
    if st.session_state.debate_active:
        render_debate_status()
        render_score_display()
        
        # Show positions
        st.markdown(f"""
        <div style='background: white; padding: 1rem; border-radius: 10px; margin: 1rem 0;'>
            <strong>Your Position:</strong> {st.session_state.user_position}<br>
            <strong>AI Position:</strong> {st.session_state.ai_position}
        </div>
        """, unsafe_allow_html=True)
        
        # Handle turns
        render_user_turn_reminder()  # Announce user's turn with voice
        render_user_input_section(controller)
        render_ai_turn(controller)
        
        # Display stored judge evaluations
        render_stored_judge_evaluations()
    
    # Show transcript
    render_debate_transcript()
    
    # Show final results
    render_final_results()
    
    # Reset button
    if st.session_state.debate_phase == 'complete':
        if st.button("Start New Debate", type="primary"):
            for key in st.session_state.keys():
                del st.session_state[key]
            st.rerun()

if __name__ == "__main__":
    main()
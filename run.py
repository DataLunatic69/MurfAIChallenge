# # Complete Enhanced AI Debate System with Streaming Audio Fix
# import os
# import json
# import random
# from typing import TypedDict, List, Dict, Literal, Optional, Annotated, Tuple
# from enum import Enum
# import operator
# from datetime import datetime
# import time
# import re
# from difflib import SequenceMatcher
# import queue
# import threading
# from concurrent.futures import ThreadPoolExecutor, as_completed
# import atexit

# # LangGraph imports
# from langgraph.graph import StateGraph, START, END
# from langgraph.checkpoint.memory import MemorySaver

# # LangChain imports
# from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
# from langchain_groq import ChatGroq

# # Voice and audio imports
# import speech_recognition as sr
# import requests
# import base64
# import io
# from pydub import AudioSegment
# from pydub.playback import play
# import tempfile

# # OpenAI imports
# import openai
# from openai import OpenAI

# # For structured output
# from pydantic import BaseModel, Field
# from langchain_core.output_parsers import PydanticOutputParser

# # For embeddings and similarity
# import numpy as np
# from typing import Deque
# from collections import deque
# # Add these imports at the top of your run.py file
# import streamlit as st
# from streamlit.runtime.scriptrunner import add_script_run_ctx
# import asyncio
# from typing import Any, Dict
# import streamlit as st
# import asyncio
# from typing import Any, Dict
# import threading

# # Add this class to handle state management for Streamlit
# class StreamlitStateManager:
#     """Manages state for Streamlit UI integration"""
    
#     def __init__(self):
#         self.current_state = {}
#         self.debate_active = False
#         self.transcript = []
#         self.scores = {"user": 0, "ai": 0}
#         self.current_round = 1
#         self.current_speaker = None
#         self.topic = ""
#         self.user_position = ""
#         self.ai_position = ""
        
#     def update_state(self, state: Dict[str, Any]):
#         """Update the current state from the debate graph"""
#         self.current_state = state
        
#         # Extract relevant information for UI
#         if "arguments" in state:
#             self.transcript = state["arguments"]
        
#         if "scores" in state:
#             user_score = sum(score.get("user_score", 0) for score in state["scores"])
#             ai_score = sum(score.get("ai_score", 0) for score in state["scores"])
#             self.scores = {"user": user_score, "ai": ai_score}
        
#         if "round_number" in state:
#             self.current_round = state["round_number"]
            
#         if "current_speaker" in state:
#             self.current_speaker = state["current_speaker"]
            
#         if "debate_active" in state:
#             self.debate_active = state["debate_active"]
            
#         if "topic" in state:
#             self.topic = state["topic"]
            
#         if "user_position" in state:
#             self.user_position = state["user_position"]
            
#         if "ai_position" in state:
#             self.ai_position = state["ai_position"]

# # Global state manager instance
# state_manager = StreamlitStateManager()

# # Modify the run_enhanced_debate function to work with Streamlit
# # def run_enhanced_debate_streamlit(topic: str = None, config: dict = None):
# #     """Run debate with Streamlit integration"""
# #     print("Starting enhanced debate with Streamlit integration...")
    
# #     initial_state = {
# #         "topic": topic, "arguments": [], "scores": [], "debate_active": True,
# #         "round_number": 1, "turn_count": 0, "debate_context": DebateContext(
# #             user_main_points=[], ai_main_points=[],
# #             unaddressed_points={"user": [], "ai": []}, evidence_used={"user": [], "ai": []}
# #         ), "repetition_warnings": [], "argument_summaries": {}
# #     }
    
# #     run_config = {"configurable": {"thread_id": f"enhanced_debate_{datetime.now().timestamp()}"}}
# #     if config: run_config.update(config)
    
# #     try:
# #         # Update state manager
# #         state_manager.update_state(initial_state)
        
# #         # Run the debate step by step for better UI control
# #         app = enhanced_debate_app
# #         config = app.checkpointer.get_tuple(run_config).config
# #         current_state = initial_state
        
# #         for step in app.stream(current_state, config, stream_mode="values"):
# #             for key, value in step.items():
# #                 current_state[key] = value
                
# #             # Update the state manager for UI
# #             state_manager.update_state(current_state)
            
# #             # Yield control to Streamlit
# #             time.sleep(0.1)
            
# #         print("Enhanced debate completed with Streamlit integration")
# #         return current_state
# #     except Exception as e:
# #         print(f"Error during debate: {e}")
# #         import traceback
# #         traceback.print_exc()
# #         return None

# # Add this function to get the current state for UI
# def get_current_state():
#     """Get the current state for UI display"""
#     return state_manager

# print("Enhanced libraries imported successfully!")

# # Load environment variables
# from dotenv import load_dotenv
# load_dotenv()

# # Configuration
# groq_api_key = os.getenv("GROQ_API_KEY")
# openai_api_key = os.getenv("OPENAI_API_KEY")
# murf_api_key = os.getenv("MURF_API_KEY")

# CONFIG = {
#     "llm_type": "groq",
#     "groq_model": "openai/gpt-oss-20b",
#     "groq_api_key": groq_api_key,
#     "openai_api_key": openai_api_key,
#     "max_rounds": 3,
#     "time_per_round": 120,
#     "debate_format": "simple",
    
#     # Feature configurations
#     "similarity_threshold": 0.85,
#     "min_argument_length": 50,
#     "enable_repetition_check": True,
#     "enable_context_awareness": True,
#     "judge_criteria_weights": {
#         "logical_coherence": 0.3,
#         "evidence_support": 0.3,
#         "relevance": 0.2,
#         "persuasiveness": 0.2
#     }
# }

# # Fixed Murf AI configuration
# MURF_CONFIG = {
#     "api_key": murf_api_key,
#     "base_url": "https://api.murf.ai/v1",
#     "voice_id": "en-US-natalie",
#     "speed": 1.0,
#     "pitch": 0.0,
#     "sample_rate": 24000,
#     "format": "wav",
#     "model": "GEN2",
#     "style": "conversational"
# }

# # Whisper configuration
# WHISPER_CONFIG = {
#     "energy_threshold": 1000,
#     "pause_threshold": 0.8,
#     "dynamic_energy_threshold": True,
#     "timeout": None,
#     "phrase_time_limit": 10
# }

# # Initialize clients
# openai_client = OpenAI(api_key=openai_api_key) if openai_api_key else None

# # Setup ffmpeg paths (optional)
# try:
#     ffmpeg_path = r"C:\ffmpeg\bin\ffmpeg.exe"
#     ffprobe_path = r"C:\ffmpeg\bin\ffprobe.exe"
#     if os.path.exists(ffmpeg_path):
#         AudioSegment.converter = ffmpeg_path
#     if os.path.exists(ffprobe_path):
#         AudioSegment.ffprobe = ffprobe_path
#         print("FFmpeg paths set explicitly")
# except:
#     print("FFmpeg not configured, using fallback methods")

# # Enums and Type Definitions
# class SpeakerType(str, Enum):
#     USER = "user"
#     AI = "ai"
#     MODERATOR = "moderator"

# class DebatePhase(str, Enum):
#     SETUP = "setup"
#     DEBATE = "debate"
#     CONCLUSION = "conclusion"
#     COMPLETE = "complete"

# class Argument(TypedDict):
#     speaker: str
#     content: str
#     timestamp: float
#     round_number: int
#     key_points: List[str]
#     rebuts_points: List[str]

# class Score(TypedDict):
#     user_score: float
#     ai_score: float
#     round_number: int
#     reasoning: str
#     detailed_scores: Dict[str, int]
#     feedback: Dict[str, List[str]]

# class DebateContext(TypedDict):
#     user_main_points: List[str]
#     ai_main_points: List[str]
#     unaddressed_points: Dict[str, List[str]]
#     evidence_used: Dict[str, List[str]]

# class DebateState(TypedDict):
#     topic: str
#     user_position: str
#     ai_position: str
#     current_speaker: str
#     first_speaker: Optional[str]
#     arguments: Annotated[List[Argument], operator.add]
#     scores: Annotated[List[Score], operator.add]
#     phase: str
#     round_number: int
#     max_rounds: int
#     turn_count: int
#     debate_active: bool
#     error_state: Optional[str]
#     current_argument: Optional[str]
#     judge_feedback: Optional[str]
#     debate_context: DebateContext
#     repetition_warnings: Annotated[List[str], operator.add]
#     argument_summaries: Dict[str, str]

# # Pydantic Models
# class JudgeScore(BaseModel):
#     logical_coherence: int = Field(ge=1, le=10, description="Score for logical coherence (1-10)")
#     evidence_support: int = Field(ge=1, le=10, description="Score for evidence and support (1-10)")
#     relevance: int = Field(ge=1, le=10, description="Score for relevance to topic (1-10)")
#     persuasiveness: int = Field(ge=1, le=10, description="Score for persuasiveness (1-10)")
#     total_score: int = Field(ge=4, le=40, description="Total score (sum of all criteria)")
#     reasoning: str = Field(description="Brief explanation of the scoring (2-3 sentences)")
#     strengths: List[str] = Field(default_factory=list, description="Key strengths of the argument")
#     weaknesses: List[str] = Field(default_factory=list, description="Areas for improvement")

# class ArgumentAnalysis(BaseModel):
#     main_claims: List[str] = Field(description="Main claims made by opponent")
#     evidence_used: List[str] = Field(description="Evidence or examples cited")
#     weak_points: List[str] = Field(description="Potential weaknesses or gaps")

# print("Structured output models defined!")

# # ============================================================================
# # STREAMING AUDIO SYSTEM
# # ============================================================================

# class AudioStreamer:
#     """Handles streaming audio playback for smooth narration."""
    
#     def __init__(self):
#         self.audio_queue = queue.Queue()
#         self.is_playing = False
#         self.playback_thread = None
#         self.stop_flag = threading.Event()
#         self.current_audio = None
    
#     def _playback_worker(self):
#         """Worker thread that continuously plays audio from the queue."""
#         while not self.stop_flag.is_set():
#             try:
#                 # Wait for audio with timeout to check stop flag
#                 audio_segment = self.audio_queue.get(timeout=0.5)
#                 if audio_segment and not self.stop_flag.is_set():
#                     self.is_playing = True
#                     play(audio_segment)
#                     self.audio_queue.task_done()
#             except queue.Empty:
#                 self.is_playing = False
#                 # No audio in queue
#                 if self.audio_queue.empty() and not self.stop_flag.is_set():
#                     time.sleep(0.1)
#             except Exception as e:
#                 print(f"Playback error: {e}")
#                 self.is_playing = False
        
#         self.is_playing = False
    
#     def start_streaming(self):
#         """Start the playback thread."""
#         if not self.playback_thread or not self.playback_thread.is_alive():
#             self.stop_flag.clear()
#             self.playback_thread = threading.Thread(target=self._playback_worker)
#             self.playback_thread.daemon = True
#             self.playback_thread.start()
    
#     def add_audio(self, audio_bytes: bytes):
#         """Add audio to the playback queue."""
#         if audio_bytes and not self.stop_flag.is_set():
#             try:
#                 audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
#                 self.audio_queue.put(audio_segment)
#             except Exception as e:
#                 print(f"Error processing audio: {e}")
    
#     def stop(self):
#         """Stop playback immediately and clear queue."""
#         self.stop_flag.set()
#         # Clear the queue
#         while not self.audio_queue.empty():
#             try:
#                 self.audio_queue.get_nowait()
#                 self.audio_queue.task_done()
#             except queue.Empty:
#                 break
#         # Wait briefly for playback to stop
#         if self.playback_thread and self.playback_thread.is_alive():
#             self.playback_thread.join(timeout=0.5)
#         self.is_playing = False
    
#     def wait_until_complete(self, timeout=30):
#         """Wait for all audio to finish playing with timeout."""
#         start_time = time.time()
#         while (not self.audio_queue.empty() or self.is_playing) and not self.stop_flag.is_set():
#             if time.time() - start_time > timeout:
#                 print("\nAudio playback timeout - stopping audio")
#                 self.stop()
#                 break
#             time.sleep(0.1)

# # Global audio streamer instance
# audio_streamer = AudioStreamer()

# class InteractiveDebateManager:
#     """Manages interactive debate flow for Streamlit UI"""
    
#     def __init__(self):
#         self.llm = get_llm()
#         self.judge_llm = get_llm()
#         self.is_processing = False
#         self.current_argument_queue = queue.Queue()
#         self.response_queue = queue.Queue()
        
#     def setup_topic(self, topic: str) -> Dict:
#         """Setup debate topic and positions"""
#         if not topic:
#             topic = "Should artificial intelligence be regulated by the government?"
        
#         prompt = f"""Given the debate topic: "{topic}"
#         What are the two main opposing positions?
#         Format: 
#         FOR: [position supporting the proposition]
#         AGAINST: [position opposing the proposition]"""
        
#         response = self.llm.invoke([HumanMessage(content=prompt)])
#         lines = response.content.strip().split('\n')
        
#         user_position = lines[0].replace("FOR:", "").strip() if lines else "In favor"
#         ai_position = lines[1].replace("AGAINST:", "").strip() if len(lines) > 1 else "Against"
        
#         return {
#             "topic": topic,
#             "user_position": user_position,
#             "ai_position": ai_position,
#             "debate_context": DebateContext(
#                 user_main_points=[], 
#                 ai_main_points=[],
#                 unaddressed_points={"user": [], "ai": []},
#                 evidence_used={"user": [], "ai": []}
#             )
#         }
    
#     def process_user_argument(self, argument: str, state: Dict) -> Tuple[bool, str, Dict]:
#         """Process user argument and return validation result"""
#         # Validate length
#         if len(argument) < CONFIG["min_argument_length"]:
#             return False, f"Argument too short. Minimum {CONFIG['min_argument_length']} characters.", {}
        
#         # Check for repetition
#         is_repetitive, similar_to = detect_repetition(
#             argument, 
#             state.get("arguments", []), 
#             SpeakerType.USER
#         )
        
#         # Extract key points
#         key_points = extract_key_points(argument, self.llm)
        
#         # Analyze what points this rebuts
#         rebuts_points = []
#         if state.get("arguments"):
#             recent_ai_args = [arg for arg in state["arguments"] if arg["speaker"] == SpeakerType.AI]
#             if recent_ai_args and CONFIG["enable_context_awareness"]:
#                 context = state.get("debate_context", {})
#                 ai_points = context.get("ai_main_points", [])
#                 for point in ai_points:
#                     if any(keyword in argument.lower() for keyword in point.lower().split()[:3]):
#                         rebuts_points.append(point)
        
#         # Create argument record
#         argument_record = Argument(
#             speaker=SpeakerType.USER,
#             content=argument,
#             timestamp=time.time(),
#             round_number=state.get("round_number", 1),
#             key_points=key_points,
#             rebuts_points=rebuts_points
#         )
        
#         # Update context
#         context = state.get("debate_context", {})
#         context["user_main_points"].extend(key_points)
        
#         return True, "Success", {
#             "argument": argument_record,
#             "context": context,
#             "is_repetitive": is_repetitive
#         }
    
#     def generate_ai_argument(self, state: Dict) -> str:
#         """Generate AI's debate argument"""
#         recent_user_args = [
#             arg for arg in state.get("arguments", []) 
#             if arg["speaker"] == SpeakerType.USER
#         ]
        
#         opponent_analysis = None
#         if recent_user_args and CONFIG["enable_context_awareness"]:
#             last_user_arg = recent_user_args[-1]
#             opponent_analysis = analyze_opponent_argument(last_user_arg["content"], self.llm)
        
#         # Build prompt
#         system_prompt = f"""You are participating in a formal debate.
# Topic: {state.get('topic', '')}
# Your position: {state.get('ai_position', '')}
# Current round: {state.get('round_number', 1)} of {state.get('max_rounds', 3)}

# IMPORTANT: Make NEW arguments, address opponent's claims, use different evidence."""
        
#         user_prompt_parts = []
#         if opponent_analysis:
#             user_prompt_parts.append(f"""Opponent's main claims:
# {chr(10).join([f"- {claim}" for claim in opponent_analysis.main_claims])}

# Weak points to address:
# {chr(10).join([f"- {weak}" for weak in opponent_analysis.weak_points])}""")
        
#         round_type = "Opening argument" if state.get('round_number', 1) == 1 else \
#                     "Rebuttal" if state.get('round_number', 1) == 2 else "Closing"
#         user_prompt_parts.append(f"\nProvide your {round_type}:")
        
#         messages = [
#             SystemMessage(content=system_prompt),
#             HumanMessage(content="\n".join(user_prompt_parts))
#         ]
        
#         response = self.llm.invoke(messages)
#         ai_argument = response.content
        
#         # Check for repetition and regenerate if needed
#         is_repetitive, _ = detect_repetition(ai_argument, state.get("arguments", []), SpeakerType.AI)
#         if is_repetitive:
#             messages.append(AIMessage(content=ai_argument))
#             messages.append(HumanMessage(content="This is too similar to previous arguments. Provide something completely different."))
#             response = self.llm.invoke(messages)
#             ai_argument = response.content
        
#         return ai_argument
    
#     def judge_argument_detailed(self, argument: str, speaker: str, state: Dict) -> JudgeScore:
#         """Judge an argument and return detailed scores"""
#         parser = PydanticOutputParser(pydantic_object=JudgeScore)
        
#         system_prompt = f"""You are an impartial debate judge. Score each criterion from 1-10.

# Scoring criteria:
# - Logical coherence: Clear reasoning and structure
# - Evidence/support: Use of facts, examples, data
# - Relevance: Directly addresses the topic
# - Persuasiveness: Compelling and convincing delivery

# {parser.get_format_instructions()}"""
        
#         position = state.get('user_position' if speaker == SpeakerType.USER else 'ai_position', '')
        
#         user_prompt = f"""Topic: {state.get('topic', '')}
# Speaker: {speaker}
# Position: {position}
# Round: {state.get('round_number', 1)}

# Argument to evaluate:
# {argument}

# Provide your evaluation:"""
        
#         messages = [
#             SystemMessage(content=system_prompt),
#             HumanMessage(content=user_prompt)
#         ]
        
#         try:
#             response = self.judge_llm.invoke(messages)
#             evaluation = parser.parse(response.content)
#         except Exception as e:
#             print(f"Judge parsing error: {e}")
#             evaluation = JudgeScore(
#                 logical_coherence=7,
#                 evidence_support=6,
#                 relevance=8,
#                 persuasiveness=7,
#                 total_score=28,
#                 reasoning="Standard evaluation applied.",
#                 strengths=["Clear argument"],
#                 weaknesses=["Could use more evidence"]
#             )
        
#         # Ensure total score is correct
#         computed_total = (
#             evaluation.logical_coherence + 
#             evaluation.evidence_support + 
#             evaluation.relevance + 
#             evaluation.persuasiveness
#         )
#         if abs(evaluation.total_score - computed_total) > 1:
#             evaluation.total_score = computed_total
        
#         return evaluation

# def speech_to_text_whisper_streamlit() -> Optional[str]:
#     """Streamlit-compatible speech to text using Whisper API"""
#     if not CONFIG["openai_api_key"]:
#         return None
    
#     recognizer = sr.Recognizer()
#     microphone = sr.Microphone()
    
#     try:
#         with microphone as source:
#             # Adjust for ambient noise
#             recognizer.adjust_for_ambient_noise(source, duration=1.0)
            
#             # Set energy threshold
#             recognizer.energy_threshold = WHISPER_CONFIG["energy_threshold"]
            
#             # Listen with timeout
#             audio = recognizer.listen(
#                 source, 
#                 timeout=WHISPER_CONFIG["timeout"],
#                 phrase_time_limit=WHISPER_CONFIG["phrase_time_limit"]
#             )
        
#         # Save to temporary file
#         with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
#             with open(tmp.name, 'wb') as f:
#                 f.write(audio.get_wav_data())
#             audio_path = tmp.name
        
#         try:
#             # Transcribe with Whisper
#             with open(audio_path, 'rb') as audio_file:
#                 transcript = openai_client.audio.transcriptions.create(
#                     model="whisper-1",
#                     file=audio_file,
#                     language="en",
#                     response_format="text"
#                 )
            
#             # Clean up
#             os.unlink(audio_path)
            
#             if transcript and transcript.strip():
#                 return transcript.strip()
#             return None
            
#         except Exception as e:
#             print(f"Transcription error: {e}")
#             if os.path.exists(audio_path):
#                 os.unlink(audio_path)
#             return None
            
#     except sr.WaitTimeoutError:
#         return None
#     except Exception as e:
#         print(f"Speech recognition error: {e}")
#         return None

# def speak_text_async(text: str, voice_id: str = None):
#     """Asynchronous version of speak_text for Streamlit"""
#     thread = threading.Thread(
#         target=speak_text_streaming,
#         args=(text, voice_id)
#     )
#     thread.daemon = True
#     thread.start()
#     return thread

# class StreamlitDebateState:
#     """Manages debate state for Streamlit UI"""
    
#     def __init__(self):
#         self.reset()
    
#     def reset(self):
#         """Reset all state variables"""
#         self.topic = ""
#         self.user_position = ""
#         self.ai_position = ""
#         self.arguments = []
#         self.scores = []
#         self.current_round = 1
#         self.max_rounds = 3
#         self.current_speaker = None
#         self.debate_active = False
#         self.debate_phase = "setup"
#         self.debate_context = None
#         self.turn_count = 0
#         self.first_speaker = None
#         self.final_winner = None
#         self.is_processing = False
    
#     def to_dict(self) -> Dict:
#         """Convert state to dictionary"""
#         return {
#             "topic": self.topic,
#             "user_position": self.user_position,
#             "ai_position": self.ai_position,
#             "arguments": self.arguments,
#             "scores": self.scores,
#             "current_round": self.current_round,
#             "max_rounds": self.max_rounds,
#             "current_speaker": self.current_speaker,
#             "debate_active": self.debate_active,
#             "debate_phase": self.debate_phase,
#             "debate_context": self.debate_context,
#             "turn_count": self.turn_count,
#             "first_speaker": self.first_speaker,
#             "final_winner": self.final_winner,
#             "is_processing": self.is_processing
#         }
    
#     def from_dict(self, data: Dict):
#         """Load state from dictionary"""
#         for key, value in data.items():
#             if hasattr(self, key):
#                 setattr(self, key, value)

# # Global instances for Streamlit
# interactive_manager = InteractiveDebateManager()
# streamlit_state = StreamlitDebateState()


# def get_interactive_manager():
#     """Get the interactive debate manager instance"""
#     global interactive_manager
#     return interactive_manager

# def get_streamlit_state():
#     """Get the streamlit state instance"""
#     global streamlit_state
#     return streamlit_state

# def reset_debate_state():
#     """Reset the debate state"""
#     global streamlit_state, audio_streamer
#     streamlit_state.reset()
#     audio_streamer.stop()

# def run_interactive_debate(topic: str = None) -> Dict:
#     """Run debate in interactive mode for Streamlit"""
#     print("Starting interactive debate session...")
    
#     manager = get_interactive_manager()
#     state = get_streamlit_state()
    
#     # Setup debate
#     setup_result = manager.setup_topic(topic)
#     state.topic = setup_result["topic"]
#     state.user_position = setup_result["user_position"]
#     state.ai_position = setup_result["ai_position"]
#     state.debate_context = setup_result["debate_context"]
#     state.debate_active = True
#     state.debate_phase = "debate"
#     state.current_round = 1
    
#     # Select first speaker
#     import random
#     state.first_speaker = random.choice([SpeakerType.USER, SpeakerType.AI])
#     state.current_speaker = state.first_speaker
    
#     print(f"Debate setup complete. First speaker: {state.first_speaker}")
#     return state.to_dict()

# # Audio cleanup on exit
# def cleanup_audio():
#     """Clean up audio resources on program exit."""
#     global audio_streamer
#     if audio_streamer:
#         audio_streamer.stop()
#         print("Audio streamer stopped")

# atexit.register(cleanup_audio)

# def cleanup_interactive_session():
#     """Clean up resources for interactive session"""
#     global audio_streamer
#     if audio_streamer:
#         audio_streamer.stop()
#     print("Interactive session cleaned up")

# # Register cleanup
# atexit.register(cleanup_interactive_session)

# # Add this function to update the state manager during the debate
# def update_state_manager(state):
#     """Update the state manager with the current state"""
#     state_manager.update_state(state)
#     # Force UI update by modifying session state
#     if 'st_session' in globals():
#         st_session = globals()['st_session']
#         if hasattr(st_session, 'session_state'):
#             st_session.session_state.last_update = time.time()

# # ============================================================================
# # ENHANCED AUDIO FUNCTIONS
# # ============================================================================

# def smart_split_text(text: str, max_chunk_size: int = 400) -> List[str]:
#     """Split text intelligently by sentences to maintain natural flow."""
#     # Split by sentences
#     sentences = re.split(r'(?<=[.!?])\s+', text)
    
#     chunks = []
#     current_chunk = ""
    
#     for sentence in sentences:
#         # If adding this sentence would exceed limit, save current chunk
#         if len(current_chunk) + len(sentence) > max_chunk_size and current_chunk:
#             chunks.append(current_chunk.strip())
#             current_chunk = sentence
#         else:
#             current_chunk += " " + sentence if current_chunk else sentence
    
#     # Add remaining text
#     if current_chunk:
#         chunks.append(current_chunk.strip())
    
#     return chunks

# def text_to_speech_murf_corrected(text: str, voice_id: str = None) -> bytes:
#     """Convert text to speech using Murf AI API with correct request format."""
#     if not MURF_CONFIG["api_key"]:
#         print("Murf API key not set.")
#         return None
    
#     # Use provided voice_id or default
#     voice_id = voice_id or MURF_CONFIG["voice_id"]
    
#     # Correct API endpoint
#     url = f"{MURF_CONFIG['base_url']}/speech/generate"
    
#     headers = {
#         "api-key": MURF_CONFIG["api_key"],
#         "Content-Type": "application/json"
#     }
    
#     # Correct request body structure based on API docs
#     payload = {
#         "text": text,
#         "voiceId": voice_id,
#         "model": MURF_CONFIG["model"],
#         "rate": MURF_CONFIG["speed"],
#         "pitch": MURF_CONFIG["pitch"],
#         "sampleRate": MURF_CONFIG["sample_rate"],
#         "format": MURF_CONFIG["format"].upper(),
#         "receiveAudioInBytes": True,
#         "style": MURF_CONFIG["style"]
#     }
    
#     try:
#         response = requests.post(url, json=payload, headers=headers, timeout=30)
        
#         if response.status_code == 200:
#             response_data = response.json()
            
#             # Handle different response formats
#             if "audioContent" in response_data:
#                 # Base64 encoded audio
#                 audio_data = base64.b64decode(response_data["audioContent"])
#                 return audio_data
#             elif "encodedAudio" in response_data and response_data["encodedAudio"]:
#                 # Base64 encoded audio (alternative field)
#                 audio_data = base64.b64decode(response_data["encodedAudio"])
#                 return audio_data
#             elif "url" in response_data:
#                 # Audio URL - download the file
#                 audio_url = response_data["url"]
#                 audio_response = requests.get(audio_url)
#                 if audio_response.status_code == 200:
#                     return audio_response.content
#                 else:
#                     print(f"Failed to download audio from URL: {audio_response.status_code}")
#                     return None
#             elif "audioFile" in response_data:
#                 # Audio file URL (Murf's actual response format)
#                 audio_url = response_data["audioFile"]
#                 audio_response = requests.get(audio_url)
#                 if audio_response.status_code == 200:
#                     return audio_response.content
#                 else:
#                     print(f"Failed to download audio from URL: {audio_response.status_code}")
#                     return None
#             else:
#                 print(f"Unexpected response format: {response_data}")
#                 return None
        
#         else:
#             print(f"Murf API error {response.status_code}: {response.text}")
            
#             # Parse error message for debugging
#             try:
#                 error_data = response.json()
#                 if "errorMessage" in error_data:
#                     print(f"Error details: {error_data['errorMessage']}")
                    
#                     # Suggest valid voice IDs if voice_id error
#                     if "voice" in error_data["errorMessage"].lower():
#                         print("Try these valid voice IDs:")
#                         print("  - en-US-natalie")
#                         print("  - en-US-ken")
#                         print("  - en-US-claire")
#                         print("  - en-UK-theo")
#                         print("  - en-AU-kylie")
#             except:
#                 pass
                
#             return None
            
#     except requests.exceptions.RequestException as e:
#         print(f"Network error: {e}")
#         return None
#     except Exception as e:
#         print(f"Murf API error: {e}")
#         return None

# def speak_text_streaming(text: str, voice_id: str = None, prefetch_count: int = 3):
#     """Convert text to speech and stream it for smooth playback using parallel fetching."""
#     print(f"Speaking (streaming): {text[:100]}...")
    
#     if not MURF_CONFIG["api_key"]:
#         print("Murf API key not set.")
#         return
    
#     chunks = smart_split_text(text, max_chunk_size=400)
#     if not chunks:
#         return
    
#     print(f"Processing {len(chunks)} chunk(s)...")
        
#     # Start the audio streamer
#     audio_streamer.start_streaming()
    
#     def fetch_chunk(chunk_data):
#         """Fetch a single chunk."""
#         index, chunk_text = chunk_data
#         audio_bytes = text_to_speech_murf_corrected(chunk_text, voice_id)
#         return index, audio_bytes
    
#     # Use ThreadPoolExecutor for parallel fetching
#     with ThreadPoolExecutor(max_workers=prefetch_count) as executor:
#         # Submit all chunks for fetching
#         futures = {}
#         for i in range(min(prefetch_count, len(chunks))):
#             future = executor.submit(fetch_chunk, (i, chunks[i]))
#             futures[future] = i
        
#         next_chunk_index = prefetch_count
#         received_chunks = {}
#         expected_index = 0
#         chunks_added = 0
        
#         while expected_index < len(chunks):
#             # Process completed futures
#             completed_futures = []
#             for future in list(futures.keys()):
#                 if future.done():
#                     try:
#                         index, audio_bytes = future.result()
#                         received_chunks[index] = audio_bytes
#                         completed_futures.append(future)
                        
#                         # Submit next chunk if available
#                         if next_chunk_index < len(chunks):
#                             new_future = executor.submit(fetch_chunk, (next_chunk_index, chunks[next_chunk_index]))
#                             futures[new_future] = next_chunk_index
#                             next_chunk_index += 1
#                     except Exception as e:
#                         print(f"Error fetching chunk: {e}")
#                         completed_futures.append(future)
            
#             # Remove completed futures
#             for future in completed_futures:
#                 del futures[future]
            
#             # Add chunks to queue in order
#             while expected_index in received_chunks:
#                 audio_bytes = received_chunks[expected_index]
#                 if audio_bytes:
#                     audio_streamer.add_audio(audio_bytes)
#                     chunks_added += 1
#                     print(f"Streaming chunk {expected_index + 1}/{len(chunks)}...", end='\r')
#                 del received_chunks[expected_index]
#                 expected_index += 1
            
#             # Small sleep to avoid busy waiting
#             time.sleep(0.05)
    
#     print(f"\nAll {chunks_added} chunks queued for streaming")

# def speak_text(text: str, voice_id: str = None):
#     """Convert text to speech and play it using streaming for smooth playback."""
#     speak_text_streaming(text, voice_id)

# def play_audio_from_bytes(audio_bytes: bytes):
#     """Play audio from raw bytes using pydub."""
#     if not audio_bytes:
#         return
        
#     try:
#         # Create AudioSegment from bytes
#         audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
#         play(audio)
#     except Exception as e:
#         print(f"Error playing audio: {e}")

# def speech_to_text_whisper_api() -> Optional[str]:
#     """Convert speech to text using OpenAI's Whisper API."""
#     if not CONFIG["openai_api_key"]:
#         print("OpenAI API key not set.")
#         return None
    
#     recognizer = sr.Recognizer()
#     microphone = sr.Microphone()
    
#     try:
#         with microphone as source:
#             print("Listening... (speak now)")
#             recognizer.adjust_for_ambient_noise(source, duration=1.0)
#             audio = recognizer.listen(source, timeout=WHISPER_CONFIG["timeout"], 
#                                     phrase_time_limit=WHISPER_CONFIG["phrase_time_limit"])
        
#         print("Processing...")
        
#         with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
#             with open(tmp.name, 'wb') as f:
#                 f.write(audio.get_wav_data())
#             audio_path = tmp.name
        
#         try:
#             with open(audio_path, 'rb') as audio_file:
#                 transcript = openai_client.audio.transcriptions.create(
#                     model="whisper-1", file=audio_file, language="en", response_format="text"
#                 )
            
#             os.unlink(audio_path)
            
#             if transcript and transcript.strip():
#                 print(f"You said: {transcript}")
#                 return transcript.strip()
#             return None
                
#         except Exception as e:
#             print(f"Transcription error: {e}")
#             return None
            
#     except sr.WaitTimeoutError:
#         print("No speech detected")
#         return None
#     except Exception as e:
#         print(f"Speech recognition error: {e}")
#         return None

# def configure_voice_settings():
#     """Configure voice settings for Murf AI."""
#     if not MURF_CONFIG["api_key"]:
#         print("Murf API key not set.")
#         return
    
#     print("\nVoice Configuration")
#     print("1. Change default voice")
#     print("2. Adjust speech speed")
#     print("3. Skip configuration")
    
#     choice = input("Select option (1-3): ")
    
#     if choice == "1":
#         new_voice = input("Enter voice ID (e.g., en-US-natalie): ")
#         MURF_CONFIG["voice_id"] = new_voice
#         print(f"Voice changed to: {new_voice}")
    
#     elif choice == "2":
#         try:
#             speed = float(input("Enter speed (0.5-2.0): "))
#             if 0.5 <= speed <= 2.0:
#                 MURF_CONFIG["speed"] = speed
#                 print(f"Speed set to: {speed}")
#             else:
#                 print("Speed must be between 0.5 and 2.0")
#         except ValueError:
#             print("Invalid speed value")

# def test_whisper_api():
#     """Test the Whisper API connection."""
#     if not CONFIG["openai_api_key"]:
#         print("OpenAI API key not set")
#         return False
    
#     try:
#         print("Testing Whisper API...")
#         return True
#     except Exception as e:
#         print(f"Whisper API test failed: {e}")
#         return False

# # ============================================================================
# # UTILITY FUNCTIONS
# # ============================================================================

# def check_similarity(text1: str, text2: str) -> float:
#     return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

# def detect_repetition(new_argument: str, previous_arguments: List[Argument], speaker: str) -> Tuple[bool, Optional[str]]:
#     if not CONFIG["enable_repetition_check"]:
#         return False, None
    
#     speaker_args = [arg for arg in previous_arguments if arg["speaker"] == speaker]
#     for prev_arg in speaker_args:
#         similarity = check_similarity(new_argument, prev_arg["content"])
#         if similarity > CONFIG["similarity_threshold"]:
#             return True, prev_arg["content"][:100] + "..."
#     return False, None

# def extract_key_points(argument: str, llm_instance) -> List[str]:
#     prompt = f"""Extract 3-5 key points from this argument. Return only the key points as a bullet list.

# Argument:
# {argument}

# Key points:"""
    
#     response = llm_instance.invoke([HumanMessage(content=prompt)])
#     points = response.content.strip().split('\n')
#     return [p.strip('- •*').strip() for p in points if p.strip()]

# def analyze_opponent_argument(argument: str, llm_instance) -> ArgumentAnalysis:
#     parser = PydanticOutputParser(pydantic_object=ArgumentAnalysis)
#     prompt = f"""Analyze this debate argument to identify main claims, evidence used, and weak points.

# {parser.get_format_instructions()}

# Argument to analyze:
# {argument}"""
    
#     response = llm_instance.invoke([HumanMessage(content=prompt)])
#     try:
#         return parser.parse(response.content)
#     except:
#         return ArgumentAnalysis(
#             main_claims=["General argument provided"],
#             evidence_used=[],
#             weak_points=["Could use more specific evidence"]
#         )

# # ============================================================================
# # LLM INITIALIZATION
# # ============================================================================

# def get_llm():
#     return ChatGroq(groq_api_key=groq_api_key, model_name="openai/gpt-oss-20b")

# llm = get_llm()
# judge_llm = get_llm()
# print(f"LLM initialized: {CONFIG['llm_type']}")

# # ============================================================================
# # DEBATE NODES
# # ============================================================================

# def topic_setup_node(state: DebateState) -> Dict:
#     print("\n=== TOPIC SETUP NODE ===")
    
#     if not state.get("topic"):
#         topic = "Should artificial intelligence be regulated by the government?"
#         user_position = "AI should be regulated"
#         ai_position = "AI should not be regulated"
#     else:
#         topic = state["topic"]
#         prompt = f"""Given the debate topic: "{topic}"
#         What are the two main opposing positions?
#         Format: 
#         FOR: [position supporting the proposition]
#         AGAINST: [position opposing the proposition]"""
        
#         response = llm.invoke([HumanMessage(content=prompt)])
#         lines = response.content.strip().split('\n')
#         user_position = lines[0].replace("FOR:", "").strip() if lines else "In favor"
#         ai_position = lines[1].replace("AGAINST:", "").strip() if len(lines) > 1 else "Against"
    
#     print(f"Topic: {topic}")
#     print(f"User Position: {user_position}")
#     print(f"AI Position: {ai_position}")
    
#     debate_context = DebateContext(
#         user_main_points=[], ai_main_points=[],
#         unaddressed_points={"user": [], "ai": []},
#         evidence_used={"user": [], "ai": []}
#     )
    
#     result = {
#         "topic": topic, "user_position": user_position, "ai_position": ai_position,
#         "phase": DebatePhase.SETUP, "debate_active": True, "max_rounds": CONFIG["max_rounds"],
#         "round_number": 1, "turn_count": 0, "debate_context": debate_context,
#         "repetition_warnings": [], "argument_summaries": {}
#     }
#     update_state_manager(result)  # Add this line
#     return result

# def voice_input_node(state: DebateState) -> Dict:
#     print("\n=== VOICE INPUT NODE ===")
    
#     # Stop any ongoing audio playback when it's user's turn
#     audio_streamer.stop()
    
#     print(f"Round {state['round_number']} - User's turn")
#     print(f"Your position: {state['user_position']}")
    
#     recent_args = [arg for arg in state.get("arguments", []) if arg["speaker"] == SpeakerType.AI]
#     if recent_args and CONFIG["enable_context_awareness"]:
#         print("\n--- Opponent's last argument ---")
#         print(recent_args[-1]["content"][:300] + "...")
#         print("--- End of opponent's argument ---\n")
    
#     # Voice input logic
#     user_argument = None
#     if CONFIG["openai_api_key"]:
#         use_voice = input("Use voice input? (y/n): ").lower() == 'y'
#         if use_voice:
#             print("\nSpeak clearly into your microphone...")
#             input("Press Enter when ready to start listening...")
#             user_argument = speech_to_text_whisper_api()
    
#     # Fallback to text input
#     if not user_argument:
#         user_argument = input("\nEnter your argument: ")
    
#     # Validation
#     valid_input = False
#     while not valid_input and user_argument:
#         if len(user_argument) < CONFIG["min_argument_length"]:
#             print(f"Argument too short. Minimum {CONFIG['min_argument_length']} characters.")
#             user_argument = input("\nEnter your argument: ")
#             continue
        
#         is_repetitive, similar_to = detect_repetition(user_argument, state.get("arguments", []), SpeakerType.USER)
#         if is_repetitive:
#             print(f"\nWARNING: Similar to previous argument: '{similar_to}'")
#             confirm = input("Submit anyway? (y/n): ")
#             if confirm.lower() != 'y':
#                 user_argument = input("\nEnter your argument: ")
#                 continue
#             else:
#                 state.get("repetition_warnings", []).append("User repeated argument")
        
#         valid_input = True
    
#     if not user_argument:
#         user_argument = "I agree with my position and believe it's the right stance."
    
#     # Process argument
#     key_points = extract_key_points(user_argument, llm)
#     rebuts_points = []
#     if recent_args and CONFIG["enable_context_awareness"]:
#         context = state.get("debate_context", {})
#         ai_points = context.get("ai_main_points", [])
#         for point in ai_points:
#             if any(keyword in user_argument.lower() for keyword in point.lower().split()[:3]):
#                 rebuts_points.append(point)
    
#     argument = Argument(
#         speaker=SpeakerType.USER, content=user_argument, timestamp=time.time(),
#         round_number=state["round_number"], key_points=key_points, rebuts_points=rebuts_points
#     )
    
#     context = state.get("debate_context", {})
#     context["user_main_points"].extend(key_points)
    
#     result = {
#         "current_argument": user_argument, "arguments": [argument],
#         "turn_count": state.get("turn_count", 0) + 1, "debate_context": context
#     }
#     update_state_manager(result)  # Add this line
#     return result

# def ai_debater_node(state: DebateState) -> Dict:
#     print("\n=== AI DEBATER NODE ===")
#     print(f"Round {state['round_number']} - AI's turn")
    
#     recent_user_args = [arg for arg in state.get("arguments", []) if arg["speaker"] == SpeakerType.USER]
#     opponent_analysis = None
#     if recent_user_args and CONFIG["enable_context_awareness"]:
#         last_user_arg = recent_user_args[-1]
#         opponent_analysis = analyze_opponent_argument(last_user_arg["content"], llm)
    
#     ai_previous = [arg["content"] for arg in state.get("arguments", []) if arg["speaker"] == SpeakerType.AI]
    
#     # Build prompt
#     system_prompt = f"""You are participating in a formal debate.
# Topic: {state['topic']}
# Your position: {state['ai_position']}
# Current round: {state['round_number']} of {state['max_rounds']}

# IMPORTANT: Make NEW arguments, address opponent's claims, use different evidence."""
    
#     user_prompt_parts = []
#     if opponent_analysis:
#         user_prompt_parts.append(f"""Opponent's main claims:
# {chr(10).join([f"- {claim}" for claim in opponent_analysis.main_claims])}

# Weak points:
# {chr(10).join([f"- {weak}" for weak in opponent_analysis.weak_points])}""")
    
#     user_prompt_parts.append(f"""
# Round {state['round_number']} task:
# {"Opening argument" if state['round_number'] == 1 else "Rebuttal" if state['round_number'] == 2 else "Closing"}

# Provide your argument:""")
    
#     messages = [SystemMessage(content=system_prompt), HumanMessage(content="\n".join(user_prompt_parts))]
#     response = llm.invoke(messages)
#     ai_argument = response.content
    
#     # Check for repetition
#     is_repetitive, _ = detect_repetition(ai_argument, state.get("arguments", []), SpeakerType.AI)
#     if is_repetitive:
#         messages.append(AIMessage(content=ai_argument))
#         messages.append(HumanMessage(content="This is too similar to previous arguments. Provide something completely different."))
#         response = llm.invoke(messages)
#         ai_argument = response.content
    
#     print(f"\nAI Argument preview: {ai_argument[:200]}...")
    
#     key_points = extract_key_points(ai_argument, llm)
#     rebuts_points = opponent_analysis.main_claims[:2] if opponent_analysis else []
    
#     argument = Argument(
#         speaker=SpeakerType.AI, content=ai_argument, timestamp=time.time(),
#         round_number=state["round_number"], key_points=key_points, rebuts_points=rebuts_points
#     )
    
#     context = state.get("debate_context", {})
#     context["ai_main_points"].extend(key_points)
    
#     result = {
#         "current_argument": ai_argument, "arguments": [argument],
#         "turn_count": state.get("turn_count", 0) + 1, "debate_context": context
#     }
#     update_state_manager(result)  # Add this line
#     return result

# def voice_output_node(state: DebateState) -> Dict:
#     """Enhanced voice output node with streaming audio."""
#     print("\n=== VOICE OUTPUT NODE (STREAMING) ===")
#     current_arg = state.get("current_argument", "")
#     current_speaker = state.get("current_speaker", "")
    
#     if current_arg and current_speaker == SpeakerType.AI and MURF_CONFIG["api_key"]:
#         print("Converting AI argument to speech with smooth streaming...")
        
#         # Use the streaming version for smooth playback
#         speak_text_streaming(current_arg)
        
#         # Wait for playback to complete
#         audio_streamer.wait_until_complete()
        
#         print("\n(Audio streaming complete)")
#     elif not MURF_CONFIG["api_key"]:
#         print("Murf API not configured - skipping voice output")
    
#     update_state_manager(state)  # Add this line
#     return state

# def judge_node(state: DebateState) -> Dict:
#     print("\n=== JUDGE NODE ===")
#     current_arg = state.get("current_argument", "")
#     current_speaker = state.get("current_speaker", "")
    
#     if not current_arg:
#         return {}
    
#     parser = PydanticOutputParser(pydantic_object=JudgeScore)
#     system_prompt = f"""You are an impartial debate judge. Score each criterion from 1-10.

# Scoring criteria:
# - Logical coherence: Clear reasoning and structure
# - Evidence/support: Use of facts, examples, data
# - Relevance: Directly addresses the topic
# - Persuasiveness: Compelling and convincing delivery

# {parser.get_format_instructions()}"""
    
#     user_prompt = f"""Topic: {state['topic']}
# Speaker: {current_speaker}
# Position: {state['user_position'] if current_speaker == SpeakerType.USER else state['ai_position']}
# Round: {state['round_number']}

# Argument to evaluate:
# {current_arg}

# Provide your evaluation:"""
    
#     messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    
#     try:
#         response = judge_llm.invoke(messages)
#         evaluation = parser.parse(response.content)
#     except Exception as e:
#         evaluation = JudgeScore(
#             logical_coherence=7, evidence_support=6, relevance=8, persuasiveness=7,
#             total_score=28, reasoning="Standard evaluation applied.",
#             strengths=["Clear argument"], weaknesses=["Could use more evidence"]
#         )
    
#     computed_total = evaluation.logical_coherence + evaluation.evidence_support + evaluation.relevance + evaluation.persuasiveness
#     if abs(evaluation.total_score - computed_total) > 1:
#         evaluation.total_score = computed_total
    
#     print(f"Judge Score: {evaluation.total_score}/40")
#     print(f"Breakdown: Logic={evaluation.logical_coherence}, Evidence={evaluation.evidence_support}, "
#           f"Relevance={evaluation.relevance}, Persuasive={evaluation.persuasiveness}")
    
#     score = Score(
#         user_score=evaluation.total_score if current_speaker == SpeakerType.USER else 0,
#         ai_score=evaluation.total_score if current_speaker == SpeakerType.AI else 0,
#         round_number=state["round_number"], reasoning=evaluation.reasoning,
#         detailed_scores={
#             "logical_coherence": evaluation.logical_coherence,
#             "evidence_support": evaluation.evidence_support,
#             "relevance": evaluation.relevance,
#             "persuasiveness": evaluation.persuasiveness
#         },
#         feedback={"strengths": evaluation.strengths, "weaknesses": evaluation.weaknesses}
#     )
    
#     result = {"scores": [score], "judge_feedback": evaluation.reasoning}
#     update_state_manager(result)  # Add this line
#     return result

# def judge_voice_feedback_node(state: DebateState) -> Dict:
#     print("\n=== VOICE JUDGE FEEDBACK NODE ===")
#     judge_result = judge_node(state)
#     feedback = judge_result.get("judge_feedback", "")
#     if feedback and MURF_CONFIG["api_key"]:
#         print("Speaking judge feedback with streaming...")
#         speak_text_streaming(f"Judge feedback: {feedback}")
#         audio_streamer.wait_until_complete()
#     update_state_manager(judge_result)  # Add this line
#     return judge_result

# def moderator_node(state: DebateState) -> Dict:
#     print("\n=== MODERATOR NODE ===")
#     current_round = state.get("round_number", 1)
#     max_rounds = state.get("max_rounds", CONFIG["max_rounds"])
#     turn_count = state.get("turn_count", 0)
#     current_speaker = state.get("current_speaker", "")
    
#     print(f"Current Round: {current_round}/{max_rounds}")
#     print(f"Turn Count: {turn_count}")
    
#     if current_round > max_rounds:
#         return {
#             "debate_active": False, "phase": DebatePhase.CONCLUSION,
#             "current_speaker": SpeakerType.MODERATOR
#         }
    
#     speakers_in_round = [arg["speaker"] for arg in state.get("arguments", []) if arg["round_number"] == current_round]
    
#     if turn_count == 0:
#         next_speaker = state.get("first_speaker", SpeakerType.USER)
#     else:
#         if current_speaker == SpeakerType.USER:
#             next_speaker = SpeakerType.AI
#         elif current_speaker == SpeakerType.AI:
#             next_speaker = SpeakerType.USER
#         else:
#             if SpeakerType.USER not in speakers_in_round:
#                 next_speaker = SpeakerType.USER
#             elif SpeakerType.AI not in speakers_in_round:
#                 next_speaker = SpeakerType.AI
#             else:
#                 next_speaker = state.get("first_speaker", SpeakerType.USER)
    
#     both_spoke = SpeakerType.USER in speakers_in_round and SpeakerType.AI in speakers_in_round
#     new_round = current_round + 1 if both_spoke else current_round
    
#     if both_spoke:
#         round_summary = f"Round {current_round}: "
#         for arg in state.get("arguments", []):
#             if arg["round_number"] == current_round:
#                 key_points = arg.get("key_points", [])
#                 round_summary += f"{arg['speaker']}: {', '.join(key_points[:2])}. "
        
#         summaries = state.get("argument_summaries", {})
#         summaries[f"round_{current_round}"] = round_summary

#     result = {
#         "current_speaker": next_speaker, "round_number": new_round,
#         "debate_active": new_round <= max_rounds,
#         "argument_summaries": summaries if 'summaries' in locals() else state.get("argument_summaries", {})
#     }
#     update_state_manager(result)  # Add this line
#     return result

# def final_judgment_node(state: DebateState) -> Dict:
#     print("\n=== FINAL JUDGMENT NODE ===")
#     all_scores = state.get("scores", [])
#     total_user_score = sum(s.get("user_score", 0) for s in all_scores)
#     total_ai_score = sum(s.get("ai_score", 0) for s in all_scores)
    
#     if total_user_score > total_ai_score:
#         winner, margin = "USER", total_user_score - total_ai_score
#     elif total_ai_score > total_user_score:
#         winner, margin = "AI", total_ai_score - total_user_score
#     else:
#         winner, margin = "TIE", 0
    
#     summary = f"""
# ╔════════════════════════════════════════════════════════════╗
# ║                    DEBATE FINAL JUDGMENT                   ║
# ╚════════════════════════════════════════════════════════════╝

# TOPIC: {state['topic']}
# FORMAT: {state.get('round_number', 0) - 1} rounds completed

# ┌────────────────────────────────────────────────────────────┐
# FINAL SCORES
# └────────────────────────────────────────────────────────────┘
# User Total: {total_user_score:.1f}/120
# AI Total:   {total_ai_score:.1f}/120

# Winner: {winner} {"(by " + str(margin) + " points)" if winner != "TIE" else ""}
# ════════════════════════════════════════════════════════════════"""
    
#     print(summary)
    
#     # Speak the final judgment if Murf is configured
#     if MURF_CONFIG["api_key"]:
#         announcement = f"Final judgment: The winner is {winner.lower()}"
#         if winner != "TIE":
#             announcement += f" by {margin} points"
#         speak_text_streaming(announcement)
#         audio_streamer.wait_until_complete()
    
#     result = {
#         "phase": DebatePhase.COMPLETE, "debate_active": False,
#         "final_summary": summary, "final_scores": {
#             "user_total": total_user_score, "ai_total": total_ai_score,
#             "winner": winner, "margin": margin
#         }
#     }
#     update_state_manager(result)  # Add this line
#     return result

# def speaker_selection_node(state: DebateState) -> Dict:
#     print("\n=== SPEAKER SELECTION NODE ===")
#     first_speaker = random.choice([SpeakerType.USER, SpeakerType.AI])
#     print(f"First speaker selected: {first_speaker}")
#     result = {"first_speaker": first_speaker, "current_speaker": first_speaker, "phase": DebatePhase.DEBATE}
#     update_state_manager(result)  # Add this line
#     return result

# def route_from_moderator(state: DebateState) -> Literal["user_input", "ai_debater", "final_judgment"]:
#     if not state.get("debate_active", True):
#         return "final_judgment"
#     next_speaker = state.get("current_speaker", "")
#     if next_speaker == SpeakerType.USER:
#         return "user_input"
#     elif next_speaker == SpeakerType.AI:
#         return "ai_debater"
#     else:
#         return "final_judgment"

# def route_after_judge(state: DebateState) -> Literal["moderator", "final_judgment"]:
#     return "moderator" if state.get("debate_active", True) else "final_judgment"

# # ============================================================================
# # GRAPH BUILDER
# # ============================================================================

# def build_voice_enhanced_debate_graph():
#     print("Building streaming voice-enhanced debate graph...")
#     workflow = StateGraph(DebateState)
    
#     workflow.add_node("topic_setup", topic_setup_node)
#     workflow.add_node("speaker_selection", speaker_selection_node)
#     workflow.add_node("user_input", voice_input_node)
#     workflow.add_node("ai_debater", ai_debater_node)
#     workflow.add_node("voice_output", voice_output_node)
#     workflow.add_node("judge", judge_voice_feedback_node)
#     workflow.add_node("moderator", moderator_node)
#     workflow.add_node("final_judgment", final_judgment_node)
    
#     workflow.add_edge(START, "topic_setup")
#     workflow.add_edge("topic_setup", "speaker_selection")
#     workflow.add_edge("speaker_selection", "moderator")
    
#     workflow.add_conditional_edges("moderator", route_from_moderator, {
#         "user_input": "user_input", "ai_debater": "ai_debater", "final_judgment": "final_judgment"
#     })
    
#     workflow.add_edge("ai_debater", "voice_output")
#     workflow.add_edge("voice_output", "judge")
#     workflow.add_edge("user_input", "judge")
    
#     workflow.add_conditional_edges("judge", route_after_judge, {
#         "moderator": "moderator", "final_judgment": "final_judgment"
#     })
    
#     workflow.add_edge("final_judgment", END)
    
#     checkpointer = MemorySaver()
#     app = workflow.compile(checkpointer=checkpointer)
#     print("Streaming voice-enhanced graph built successfully!")
#     return app

# enhanced_debate_app = build_voice_enhanced_debate_graph()

# # ============================================================================
# # MAIN EXECUTION
# # ============================================================================

# def run_enhanced_debate(topic: str = None, config: dict = None):
#     print("=" * 60)
#     print("STARTING ENHANCED DEBATE SESSION WITH STREAMING")
#     print("=" * 60)
    
#     initial_state = {
#         "topic": topic, "arguments": [], "scores": [], "debate_active": True,
#         "round_number": 1, "turn_count": 0, "debate_context": DebateContext(
#             user_main_points=[], ai_main_points=[],
#             unaddressed_points={"user": [], "ai": []}, evidence_used={"user": [], "ai": []}
#         ), "repetition_warnings": [], "argument_summaries": {}
#     }
    
#     run_config = {"configurable": {"thread_id": f"enhanced_debate_{datetime.now().timestamp()}"}}
#     if config: run_config.update(config)
    
#     try:
#         final_state = enhanced_debate_app.invoke(initial_state, run_config)
#         print("\n" + "=" * 60)
#         print("ENHANCED DEBATE COMPLETED")
#         print("=" * 60)
#         return final_state
#     except Exception as e:
#         print(f"Error during debate: {e}")
#         import traceback
#         traceback.print_exc()
#         return None

# def display_enhanced_transcript(state: dict):
#     print("\n" + "=" * 60)
#     print("ENHANCED DEBATE TRANSCRIPT")
#     print("=" * 60)
    
#     print(f"\nTopic: {state.get('topic', 'No topic')}")
#     print(f"User Position: {state.get('user_position', '')}")
#     print(f"AI Position: {state.get('ai_position', '')}")
#     print("\n" + "-" * 60)
    
#     arguments = state.get("arguments", [])
#     for i, arg in enumerate(arguments):
#         print(f"\n{'='*60}")
#         print(f"Round {arg['round_number']} - {arg['speaker'].upper()}")
#         print(f"{'='*60}")
        
#         if arg.get("key_points"):
#             print("\nKey Points:")
#             for point in arg["key_points"][:3]: 
#                 print(f"  • {point}")
        
#         print(f"\nArgument:")
#         print(arg['content'][:400])
#         if len(arg['content']) > 400: 
#             print("...")
    
#     if state.get("final_summary"):
#         print("\n" + state["final_summary"])

# def test_murf_integration():
#     """Test the streaming Murf integration."""
#     print("\nTesting Streaming Murf AI integration...")
    
#     if not MURF_CONFIG["api_key"]:
#         print("MURF_API_KEY not set in environment variables")
#         return False
    
#     # Test with a shorter phrase for quick testing
#     test_text = "Hello, this is a test of the Murf AI streaming system."
    
#     print(f"Testing streaming with text: '{test_text}'")
    
#     try:
#         # Generate and play audio
#         speak_text_streaming(test_text)
        
#         # Wait for playback with timeout
#         print("Waiting for playback to complete...")
#         audio_streamer.wait_until_complete(timeout=10)
        
#         print("Streaming audio test successful!")
#         return True
#     except Exception as e:
#         print(f"Streaming test failed: {e}")
#         return False
    

# def run_enhanced_debate_streamlit(topic: str = None):
#     """Run debate with Streamlit integration"""
#     print("Starting enhanced debate with Streamlit integration...")
    
#     initial_state = {
#         "topic": topic, "arguments": [], "scores": [], "debate_active": True,
#         "round_number": 1, "turn_count": 0, "debate_context": DebateContext(
#             user_main_points=[], ai_main_points=[],
#             unaddressed_points={"user": [], "ai": []}, evidence_used={"user": [], "ai": []}
#         ), "repetition_warnings": [], "argument_summaries": {}
#     }
    
#     # Update state manager
#     state_manager.update_state(initial_state)
    
#     try:
#         # Create a new memory saver for this session
#         checkpointer = MemorySaver()
        
#         # Build the app with the checkpointer
#         workflow = StateGraph(DebateState)
        
#         workflow.add_node("topic_setup", topic_setup_node)
#         workflow.add_node("speaker_selection", speaker_selection_node)
#         workflow.add_node("user_input", voice_input_node)
#         workflow.add_node("ai_debater", ai_debater_node)
#         workflow.add_node("voice_output", voice_output_node)
#         workflow.add_node("judge", judge_voice_feedback_node)
#         workflow.add_node("moderator", moderator_node)
#         workflow.add_node("final_judgment", final_judgment_node)
        
#         workflow.add_edge(START, "topic_setup")
#         workflow.add_edge("topic_setup", "speaker_selection")
#         workflow.add_edge("speaker_selection", "moderator")
        
#         workflow.add_conditional_edges("moderator", route_from_moderator, {
#             "user_input": "user_input", "ai_debater": "ai_debater", "final_judgment": "final_judgment"
#         })
        
#         workflow.add_edge("ai_debater", "voice_output")
#         workflow.add_edge("voice_output", "judge")
#         workflow.add_edge("user_input", "judge")
        
#         workflow.add_conditional_edges("judge", route_after_judge, {
#             "moderator": "moderator", "final_judgment": "final_judgment"
#         })
        
#         workflow.add_edge("final_judgment", END)
        
#         app = workflow.compile(checkpointer=checkpointer)
        
#         # Run the debate with a unique thread ID
#         run_config = {"configurable": {"thread_id": f"streamlit_debate_{datetime.now().timestamp()}"}}
        
#         # Run the debate
#         final_state = app.invoke(initial_state, run_config)
        
#         # Update the state manager with final state
#         state_manager.update_state(final_state)
#         state_manager.debate_active = False
        
#         print("Enhanced debate completed with Streamlit integration")
#         return final_state
        
#     except Exception as e:
#         print(f"Error during debate: {e}")
#         import traceback
#         traceback.print_exc()
#         state_manager.debate_active = False
#         return None

# # Add this function to get the current state for UI
# def get_current_state():
#     """Get the current state for UI display"""
#     return state_manager

# # Add this function to run debate in a thread
# def run_debate_in_thread(topic):
#     """Run debate in a separate thread"""
#     run_enhanced_debate_streamlit(topic)

# # ============================================================================
# # MAIN PROGRAM
# # ============================================================================

# if __name__ == "__main__":
#     test_topics = [
#         "Should social media platforms be held responsible for user-generated content?",
#         "Is artificial general intelligence an existential threat to humanity?",
#         "Should governments implement universal basic income?",
#         "Is remote work better than office work for productivity?",
#     ]
    
#     print("ENHANCED DEBATE SYSTEM WITH STREAMING VOICE")
#     print("=" * 60)
    
#     # Test APIs
#     if CONFIG["openai_api_key"]:
#         print("✓ OpenAI API configured")
#     else:
#         print("✗ OPENAI_API_KEY not set. Voice input disabled.")
    
#     if MURF_CONFIG["api_key"]:
#         print("✓ Murf API configured (with streaming support)")
#         configure_voice = input("Configure voice output settings? (y/n): ").lower() == 'y'
#         if configure_voice:
#             configure_voice_settings()
        
#         # Test Murf integration with streaming
#         test_tts = input("Test streaming text-to-speech? (y/n): ").lower() == 'y'
#         if test_tts:
#             test_murf_integration()
#     else:
#         print("✗ MURF_API_KEY not set. Voice output disabled.")
    
#     # Topic selection
#     print("\nChoose a topic:")
#     for i, topic in enumerate(test_topics, 1):
#         print(f"{i}. {topic}")
#     print("5. Custom topic")
    
#     choice = input("\nSelect (1-5): ")
#     if choice == "5":
#         debate_topic = input("Enter your custom topic: ")
#     elif choice in ["1", "2", "3", "4"]:
#         debate_topic = test_topics[int(choice) - 1]
#     else:
#         debate_topic = test_topics[0]
    
#     print(f"\nSelected Topic: {debate_topic}")
#     print("\nPress Enter to begin...")
#     input()
    
#     # Run debate
#     final_state = run_enhanced_debate(topic=debate_topic)
    
#     # Display results
#     if final_state:
#         display_enhanced_transcript(final_state)
#         save_option = input("\nSave transcript? (y/n): ")
#         if save_option.lower() == 'y':
#             filename = f"debate_transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
#             with open(filename, 'w') as f:
#                 save_state = {k: v for k, v in final_state.items() if k != 'debate_context'}
#                 json.dump(save_state, f, indent=2, default=str)
#             print(f"Transcript saved to {filename}")
    
#     # Clean up audio resources
#     cleanup_audio()
    
#     print("\nDebate session complete!")


# Complete Enhanced AI Debate System with Streaming Audio and Streamlit Integration
import os
import json
import random
from typing import TypedDict, List, Dict, Literal, Optional, Annotated, Tuple, Any
from enum import Enum
import operator
from datetime import datetime
import time
import re
from difflib import SequenceMatcher
import queue
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import atexit
import tempfile

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# LangChain imports
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq

# Voice and audio imports
import speech_recognition as sr
import requests
import base64
import io
from pydub import AudioSegment
from pydub.playback import play

# OpenAI imports
import openai
from openai import OpenAI

# For structured output
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

# For embeddings and similarity
import numpy as np
from typing import Deque
from collections import deque

print("Enhanced libraries imported successfully!")

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configuration
groq_api_key = os.getenv("GROQ_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
murf_api_key = os.getenv("MURF_API_KEY")

CONFIG = {
    "llm_type": "groq",
    "groq_model": "llama-3.3-70b-versatile",
    "groq_api_key": groq_api_key,
    "openai_api_key": openai_api_key,
    "max_rounds": 3,
    "time_per_round": 120,
    "debate_format": "simple",
    
    # Feature configurations
    "similarity_threshold": 0.85,
    "min_argument_length": 50,
    "enable_repetition_check": True,
    "enable_context_awareness": True,
    "judge_criteria_weights": {
        "logical_coherence": 0.3,
        "evidence_support": 0.3,
        "relevance": 0.2,
        "persuasiveness": 0.2
    }
}

# Fixed Murf AI configuration
MURF_CONFIG = {
    "api_key": murf_api_key,
    "base_url": "https://api.murf.ai/v1",
    "voice_id": "en-US-natalie",
    "speed": 1.0,
    "pitch": 0.0,
    "sample_rate": 24000,
    "format": "wav",
    "model": "GEN2",
    "style": "conversational"
}

# Whisper configuration
WHISPER_CONFIG = {
    "energy_threshold": 1000,
    "pause_threshold": 0.8,
    "dynamic_energy_threshold": True,
    "timeout": None,
    "phrase_time_limit": 10
}

def get_llm():
    """Return a ChatGroq LLM instance using the configured model and API key."""
    return ChatGroq(
        groq_api_key=CONFIG["groq_api_key"],
        model_name=CONFIG.get("groq_model", "llama-3.3-70b-versatile")
    )

# Initialize clients
openai_client = OpenAI(api_key=openai_api_key) if openai_api_key else None


# Setup ffmpeg paths (optional)
try:
    ffmpeg_path = r"C:\ffmpeg\bin\ffmpeg.exe"
    ffprobe_path = r"C:\ffmpeg\bin\ffprobe.exe"
    if os.path.exists(ffmpeg_path):
        AudioSegment.converter = ffmpeg_path
    if os.path.exists(ffprobe_path):
        AudioSegment.ffprobe = ffprobe_path
        print("FFmpeg paths set explicitly")
except:
    print("FFmpeg not configured, using fallback methods")

# Enums and Type Definitions
class SpeakerType(str, Enum):
    USER = "user"
    AI = "ai"
    MODERATOR = "moderator"

class DebatePhase(str, Enum):
    SETUP = "setup"
    DEBATE = "debate"
    CONCLUSION = "conclusion"
    COMPLETE = "complete"

class Argument(TypedDict):
    speaker: str
    content: str
    timestamp: float
    round_number: int
    key_points: List[str]
    rebuts_points: List[str]

class Score(TypedDict):
    user_score: float
    ai_score: float
    round_number: int
    reasoning: str
    detailed_scores: Dict[str, int]
    feedback: Dict[str, List[str]]

class DebateContext(TypedDict):
    user_main_points: List[str]
    ai_main_points: List[str]
    unaddressed_points: Dict[str, List[str]]
    evidence_used: Dict[str, List[str]]

class DebateState(TypedDict):
    topic: str
    user_position: str
    ai_position: str
    current_speaker: str
    first_speaker: Optional[str]
    arguments: Annotated[List[Argument], operator.add]
    scores: Annotated[List[Score], operator.add]
    phase: str
    round_number: int
    max_rounds: int
    turn_count: int
    debate_active: bool
    error_state: Optional[str]
    current_argument: Optional[str]
    judge_feedback: Optional[str]
    debate_context: DebateContext
    repetition_warnings: Annotated[List[str], operator.add]
    argument_summaries: Dict[str, str]

# Pydantic Models
class JudgeScore(BaseModel):
    logical_coherence: int = Field(ge=1, le=10, description="Score for logical coherence (1-10)")
    evidence_support: int = Field(ge=1, le=10, description="Score for evidence and support (1-10)")
    relevance: int = Field(ge=1, le=10, description="Score for relevance to topic (1-10)")
    persuasiveness: int = Field(ge=1, le=10, description="Score for persuasiveness (1-10)")
    total_score: int = Field(ge=4, le=40, description="Total score (sum of all criteria)")
    reasoning: str = Field(description="Brief explanation of the scoring (2-3 sentences)")
    strengths: List[str] = Field(default_factory=list, description="Key strengths of the argument")
    weaknesses: List[str] = Field(default_factory=list, description="Areas for improvement")

class ArgumentAnalysis(BaseModel):
    main_claims: List[str] = Field(description="Main claims made by opponent")
    evidence_used: List[str] = Field(description="Evidence or examples cited")
    weak_points: List[str] = Field(description="Potential weaknesses or gaps")

print("Structured output models defined!")

# ============================================================================
# STREAMING AUDIO SYSTEM
# ============================================================================

class AudioStreamer:
    """Handles streaming audio playback for smooth narration."""
    
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.is_playing = False
        self.playback_thread = None
        self.stop_flag = threading.Event()
        self.current_audio = None
    
    def _playback_worker(self):
        """Worker thread that continuously plays audio from the queue."""
        while not self.stop_flag.is_set():
            try:
                # Wait for audio with timeout to check stop flag
                audio_segment = self.audio_queue.get(timeout=0.5)
                if audio_segment and not self.stop_flag.is_set():
                    self.is_playing = True
                    play(audio_segment)
                    self.audio_queue.task_done()
            except queue.Empty:
                self.is_playing = False
                # No audio in queue
                if self.audio_queue.empty() and not self.stop_flag.is_set():
                    time.sleep(0.1)
            except Exception as e:
                print(f"Playback error: {e}")
                self.is_playing = False
        
        self.is_playing = False
    
    def start_streaming(self):
        """Start the playback thread."""
        if not self.playback_thread or not self.playback_thread.is_alive():
            self.stop_flag.clear()
            self.playback_thread = threading.Thread(target=self._playback_worker)
            self.playback_thread.daemon = True
            self.playback_thread.start()
    
    def add_audio(self, audio_bytes: bytes):
        """Add audio to the playback queue."""
        if audio_bytes and not self.stop_flag.is_set():
            try:
                audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
                self.audio_queue.put(audio_segment)
            except Exception as e:
                print(f"Error processing audio: {e}")
    
    def stop(self):
        """Stop playback immediately and clear queue."""
        self.stop_flag.set()
        # Clear the queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.task_done()
            except queue.Empty:
                break
        # Wait briefly for playback to stop
        if self.playback_thread and self.playback_thread.is_alive():
            self.playback_thread.join(timeout=0.5)
        self.is_playing = False
    
    def wait_until_complete(self, timeout=30):
        """Wait for all audio to finish playing with timeout."""
        start_time = time.time()
        while (not self.audio_queue.empty() or self.is_playing) and not self.stop_flag.is_set():
            if time.time() - start_time > timeout:
                print("\nAudio playback timeout - stopping audio")
                self.stop()
                break
            time.sleep(0.1)

# Global audio streamer instance
audio_streamer = AudioStreamer()

# Audio cleanup on exit
def cleanup_audio():
    """Clean up audio resources on program exit."""
    global audio_streamer
    if audio_streamer:
        audio_streamer.stop()
        print("Audio streamer stopped")

atexit.register(cleanup_audio)

# ============================================================================
# ENHANCED AUDIO FUNCTIONS
# ============================================================================

def smart_split_text(text: str, max_chunk_size: int = 400) -> List[str]:
    """Split text intelligently by sentences to maintain natural flow."""
    # Split by sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # If adding this sentence would exceed limit, save current chunk
        if len(current_chunk) + len(sentence) > max_chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += " " + sentence if current_chunk else sentence
    
    # Add remaining text
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def text_to_speech_murf_corrected(text: str, voice_id: str = None) -> bytes:
    """Convert text to speech using Murf AI API with correct request format."""
    if not MURF_CONFIG["api_key"]:
        print("Murf API key not set.")
        return None
    
    # Use provided voice_id or default
    voice_id = voice_id or MURF_CONFIG["voice_id"]
    
    # Correct API endpoint
    url = f"{MURF_CONFIG['base_url']}/speech/generate"
    
    headers = {
        "api-key": MURF_CONFIG["api_key"],
        "Content-Type": "application/json"
    }
    
    # Correct request body structure based on API docs
    payload = {
        "text": text,
        "voiceId": voice_id,
        "model": MURF_CONFIG["model"],
        "rate": MURF_CONFIG["speed"],
        "pitch": MURF_CONFIG["pitch"],
        "sampleRate": MURF_CONFIG["sample_rate"],
        "format": MURF_CONFIG["format"].upper(),
        "receiveAudioInBytes": True,
        "style": MURF_CONFIG["style"]
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        
        if response.status_code == 200:
            response_data = response.json()
            
            # Handle different response formats
            if "audioContent" in response_data:
                # Base64 encoded audio
                audio_data = base64.b64decode(response_data["audioContent"])
                return audio_data
            elif "encodedAudio" in response_data and response_data["encodedAudio"]:
                # Base64 encoded audio (alternative field)
                audio_data = base64.b64decode(response_data["encodedAudio"])
                return audio_data
            elif "url" in response_data:
                # Audio URL - download the file
                audio_url = response_data["url"]
                audio_response = requests.get(audio_url)
                if audio_response.status_code == 200:
                    return audio_response.content
                else:
                    print(f"Failed to download audio from URL: {audio_response.status_code}")
                    return None
            elif "audioFile" in response_data:
                # Audio file URL (Murf's actual response format)
                audio_url = response_data["audioFile"]
                audio_response = requests.get(audio_url)
                if audio_response.status_code == 200:
                    return audio_response.content
                else:
                    print(f"Failed to download audio from URL: {audio_response.status_code}")
                    return None
            else:
                print(f"Unexpected response format: {response_data}")
                return None
        
        else:
            print(f"Murf API error {response.status_code}: {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"Network error: {e}")
        return None
    except Exception as e:
        print(f"Murf API error: {e}")
        return None

def speak_text_streaming(text: str, voice_id: str = None, prefetch_count: int = 3):
    """Convert text to speech and stream it for smooth playback using parallel fetching."""
    print(f"Speaking (streaming): {text[:100]}...")
    
    if not MURF_CONFIG["api_key"]:
        print("Murf API key not set.")
        return
    
    chunks = smart_split_text(text, max_chunk_size=400)
    if not chunks:
        return
    
    print(f"Processing {len(chunks)} chunk(s)...")
        
    # Start the audio streamer
    audio_streamer.start_streaming()
    
    def fetch_chunk(chunk_data):
        """Fetch a single chunk."""
        index, chunk_text = chunk_data
        audio_bytes = text_to_speech_murf_corrected(chunk_text, voice_id)
        return index, audio_bytes
    
    # Use ThreadPoolExecutor for parallel fetching
    with ThreadPoolExecutor(max_workers=prefetch_count) as executor:
        # Submit all chunks for fetching
        futures = {}
        for i in range(min(prefetch_count, len(chunks))):
            future = executor.submit(fetch_chunk, (i, chunks[i]))
            futures[future] = i
        
        next_chunk_index = prefetch_count
        received_chunks = {}
        expected_index = 0
        chunks_added = 0
        
        while expected_index < len(chunks):
            # Process completed futures
            completed_futures = []
            for future in list(futures.keys()):
                if future.done():
                    try:
                        index, audio_bytes = future.result()
                        received_chunks[index] = audio_bytes
                        completed_futures.append(future)
                        
                        # Submit next chunk if available
                        if next_chunk_index < len(chunks):
                            new_future = executor.submit(fetch_chunk, (next_chunk_index, chunks[next_chunk_index]))
                            futures[new_future] = next_chunk_index
                            next_chunk_index += 1
                    except Exception as e:
                        print(f"Error fetching chunk: {e}")
                        completed_futures.append(future)
            
            # Remove completed futures
            for future in completed_futures:
                del futures[future]
            
            # Add chunks to queue in order
            while expected_index in received_chunks:
                audio_bytes = received_chunks[expected_index]
                if audio_bytes:
                    audio_streamer.add_audio(audio_bytes)
                    chunks_added += 1
                    print(f"Streaming chunk {expected_index + 1}/{len(chunks)}...", end='\r')
                del received_chunks[expected_index]
                expected_index += 1
            
            # Small sleep to avoid busy waiting
            time.sleep(0.05)
    
    print(f"\nAll {chunks_added} chunks queued for streaming")

def speak_text(text: str, voice_id: str = None):
    """Convert text to speech and play it using streaming for smooth playback."""
    speak_text_streaming(text, voice_id)

def speech_to_text_whisper_api() -> Optional[str]:
    """Convert speech to text using OpenAI's Whisper API."""
    if not CONFIG["openai_api_key"]:
        print("OpenAI API key not set.")
        return None
    
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    
    try:
        with microphone as source:
            print("Listening... (speak now)")
            recognizer.adjust_for_ambient_noise(source, duration=1.0)
            audio = recognizer.listen(source, timeout=WHISPER_CONFIG["timeout"], 
                                    phrase_time_limit=WHISPER_CONFIG["phrase_time_limit"])
        
        print("Processing...")
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            with open(tmp.name, 'wb') as f:
                f.write(audio.get_wav_data())
            audio_path = tmp.name
        
        try:
            with open(audio_path, 'rb') as audio_file:
                transcript = openai_client.audio.transcriptions.create(
                    model="whisper-1", file=audio_file, language="en", response_format="text"
                )
            
            os.unlink(audio_path)
            
            if transcript and transcript.strip():
                print(f"You said: {transcript}")
                return transcript.strip()
            return None
                
        except Exception as e:
            print(f"Transcription error: {e}")
            return None
            
    except sr.WaitTimeoutError:
        print("No speech detected")
        return None
    except Exception as e:
        print(f"Speech recognition error: {e}")
        return None

def speak_text_async(text: str, voice_id: str = None):
    """Asynchronous version of speak_text for Streamlit"""
    thread = threading.Thread(
        target=speak_text_streaming,
        args=(text, voice_id)
    )
    thread.daemon = True
    thread.start()
    return thread

# ============================================================================
# STREAMLIT STATE SYNCHRONIZATION
# ============================================================================

class StreamlitDebateState:
    """Manages debate state for Streamlit UI"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all state variables"""
        self.topic = ""
        self.user_position = ""
        self.ai_position = ""
        self.arguments = []
        self.scores = []
        self.current_round = 1
        self.max_rounds = 3
        self.current_speaker = None
        self.debate_active = False
        self.debate_phase = "setup"
        self.debate_context = None
        self.turn_count = 0
        self.first_speaker = None
        self.final_winner = None
        self.is_processing = False
    
    def to_dict(self) -> Dict:
        """Convert state to dictionary"""
        return {
            "topic": self.topic,
            "user_position": self.user_position,
            "ai_position": self.ai_position,
            "arguments": self.arguments,
            "scores": self.scores,
            "current_round": self.current_round,
            "max_rounds": self.max_rounds,
            "current_speaker": self.current_speaker,
            "debate_active": self.debate_active,
            "debate_phase": self.debate_phase,
            "debate_context": self.debate_context,
            "turn_count": self.turn_count,
            "first_speaker": self.first_speaker,
            "final_winner": self.final_winner,
            "is_processing": self.is_processing
        }
    
    def from_dict(self, data: Dict):
        """Load state from dictionary"""
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)


class InteractiveDebateManager:
    """Manages interactive debate flow for Streamlit UI"""
    
    def __init__(self):
        self.llm = get_llm()
        self.judge_llm = get_llm()
        self.is_processing = False
        self.current_argument_queue = queue.Queue()
        self.response_queue = queue.Queue()
        
    def setup_topic(self, topic: str) -> Dict:
        """Setup debate topic and positions"""
        if not topic:
            topic = "Should artificial intelligence be regulated by the government?"
        
        prompt = f"""Given the debate topic: "{topic}"
        What are the two main opposing positions?
        Format: 
        FOR: [position supporting the proposition]
        AGAINST: [position opposing the proposition]"""
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        lines = response.content.strip().split('\n')
        
        user_position = lines[0].replace("FOR:", "").strip() if lines else "In favor"
        ai_position = lines[1].replace("AGAINST:", "").strip() if len(lines) > 1 else "Against"
        
        return {
            "topic": topic,
            "user_position": user_position,
            "ai_position": ai_position,
            "debate_context": DebateContext(
                user_main_points=[], 
                ai_main_points=[],
                unaddressed_points={"user": [], "ai": []},
                evidence_used={"user": [], "ai": []}
            )
        }
    
    def process_user_argument(self, argument: str, state: Dict) -> Tuple[bool, str, Dict]:
        """Process user argument and return validation result"""
        # Validate length
        if len(argument) < CONFIG["min_argument_length"]:
            return False, f"Argument too short. Minimum {CONFIG['min_argument_length']} characters.", {}
        
        # Check for repetition
        is_repetitive, similar_to = detect_repetition(
            argument, 
            state.get("arguments", []), 
            SpeakerType.USER
        )
        
        # Extract key points
        key_points = extract_key_points(argument, self.llm)
        
        # Analyze what points this rebuts
        rebuts_points = []
        if state.get("arguments"):
            recent_ai_args = [arg for arg in state["arguments"] if arg["speaker"] == SpeakerType.AI]
            if recent_ai_args and CONFIG["enable_context_awareness"]:
                context = state.get("debate_context", {})
                ai_points = context.get("ai_main_points", [])
                for point in ai_points:
                    if any(keyword in argument.lower() for keyword in point.lower().split()[:3]):
                        rebuts_points.append(point)
        
        # Create argument record
        argument_record = Argument(
            speaker=SpeakerType.USER,
            content=argument,
            timestamp=time.time(),
            round_number=state.get("round_number", 1),
            key_points=key_points,
            rebuts_points=rebuts_points
        )
        
        # Update context
        context = state.get("debate_context", {})
        context["user_main_points"].extend(key_points)
        
        return True, "Success", {
            "argument": argument_record,
            "context": context,
            "is_repetitive": is_repetitive
        }
    
    def generate_ai_argument(self, state: Dict) -> str:
        """Generate AI's debate argument"""
        recent_user_args = [
            arg for arg in state.get("arguments", []) 
            if arg["speaker"] == SpeakerType.USER
        ]
        
        opponent_analysis = None
        if recent_user_args and CONFIG["enable_context_awareness"]:
            last_user_arg = recent_user_args[-1]
            opponent_analysis = analyze_opponent_argument(last_user_arg["content"], self.llm)
        
        # Build prompt
        system_prompt = f"""You are participating in a formal debate.
Topic: {state.get('topic', '')}
Your position: {state.get('ai_position', '')}
Current round: {state.get('round_number', 1)} of {state.get('max_rounds', 3)}

IMPORTANT: Make NEW arguments, address opponent's claims, use different evidence."""
        
        user_prompt_parts = []
        if opponent_analysis:
            user_prompt_parts.append(f"""Opponent's main claims:
{chr(10).join([f"- {claim}" for claim in opponent_analysis.main_claims])}

Weak points to address:
{chr(10).join([f"- {weak}" for weak in opponent_analysis.weak_points])}""")
        
        round_type = "Opening argument" if state.get('round_number', 1) == 1 else \
                    "Rebuttal" if state.get('round_number', 1) == 2 else "Closing"
        user_prompt_parts.append(f"\nProvide your {round_type}:")
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content="\n".join(user_prompt_parts))
        ]
        
        response = self.llm.invoke(messages)
        ai_argument = response.content
        
        # Check for repetition and regenerate if needed
        is_repetitive, _ = detect_repetition(ai_argument, state.get("arguments", []), SpeakerType.AI)
        if is_repetitive:
            messages.append(AIMessage(content=ai_argument))
            messages.append(HumanMessage(content="This is too similar to previous arguments. Provide something completely different."))
            response = self.llm.invoke(messages)
            ai_argument = response.content
        
        return ai_argument
    
    def judge_argument_detailed(self, argument: str, speaker: str, state: Dict) -> JudgeScore:
        """Judge an argument and return detailed scores"""
        parser = PydanticOutputParser(pydantic_object=JudgeScore)
        
        system_prompt = f"""You are an impartial debate judge. Score each criterion from 1-10.

Scoring criteria:
- Logical coherence: Clear reasoning and structure
- Evidence/support: Use of facts, examples, data
- Relevance: Directly addresses the topic
- Persuasiveness: Compelling and convincing delivery

{parser.get_format_instructions()}"""
        
        position = state.get('user_position' if speaker == SpeakerType.USER else 'ai_position', '')
        
        user_prompt = f"""Topic: {state.get('topic', '')}
Speaker: {speaker}
Position: {position}
Round: {state.get('round_number', 1)}

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
        except Exception as e:
            print(f"Judge parsing error: {e}")
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
        
        # Ensure total score is correct
        computed_total = (
            evaluation.logical_coherence + 
            evaluation.evidence_support + 
            evaluation.relevance + 
            evaluation.persuasiveness
        )
        if abs(evaluation.total_score - computed_total) > 1:
            evaluation.total_score = computed_total
        
        return evaluation


# Global instances for Streamlit
interactive_manager = InteractiveDebateManager()
streamlit_state = StreamlitDebateState()

# ============================================================================
# STREAMLIT STATE MANAGER (LEGACY COMPATIBILITY)
# ============================================================================

class StreamlitStateManager:
    """Legacy state manager for backward compatibility"""
    
    def __init__(self):
        self.current_state = {}
        self.debate_active = False
        self.transcript = []
        self.scores = {"user": 0, "ai": 0}
        self.current_round = 1
        self.current_speaker = None
        self.topic = ""
        self.user_position = ""
        self.ai_position = ""
        
    def update_state(self, state: Dict[str, Any]):
        """Update the current state from the debate graph"""
        self.current_state = state
        
        if "arguments" in state:
            self.transcript = state["arguments"]
        
        if "scores" in state:
            user_score = sum(score.get("user_score", 0) for score in state["scores"])
            ai_score = sum(score.get("ai_score", 0) for score in state["scores"])
            self.scores = {"user": user_score, "ai": ai_score}
        
        if "round_number" in state:
            self.current_round = state["round_number"]
            
        if "current_speaker" in state:
            self.current_speaker = state["current_speaker"]
            
        if "debate_active" in state:
            self.debate_active = state["debate_active"]
            
        if "topic" in state:
            self.topic = state["topic"]
            
        if "user_position" in state:
            self.user_position = state["user_position"]
            
        if "ai_position" in state:
            self.ai_position = state["ai_position"]

# Global state manager instance (legacy)
state_manager = StreamlitStateManager()

# ============================================================================
# EXPORT FUNCTIONS FOR STREAMLIT
# ============================================================================

def get_interactive_manager():
    """Get the interactive debate manager instance"""
    global interactive_manager
    return interactive_manager

def get_streamlit_state():
    """Get the streamlit state instance"""
    global streamlit_state
    return streamlit_state

def get_current_state():
    """Get the current state for UI display (legacy)"""
    return state_manager

def reset_debate_state():
    """Reset the debate state"""
    global streamlit_state, audio_streamer
    streamlit_state.reset()
    audio_streamer.stop()

# ============================================================================
# INTERACTIVE DEBATE RUNNER
# ============================================================================

def run_interactive_debate(topic: str = None) -> Dict:
    """Run debate in interactive mode for Streamlit"""
    print("Starting interactive debate session...")
    
    manager = get_interactive_manager()
    state = get_streamlit_state()
    
    # Setup debate
    setup_result = manager.setup_topic(topic)
    state.topic = setup_result["topic"]
    state.user_position = setup_result["user_position"]
    state.ai_position = setup_result["ai_position"]
    state.debate_context = setup_result["debate_context"]
    state.debate_active = True
    state.debate_phase = "debate"
    state.current_round = 1
    
    # Select first speaker
    import random
    state.first_speaker = random.choice([SpeakerType.USER, SpeakerType.AI])
    state.current_speaker = state.first_speaker
    
    print(f"Debate setup complete. First speaker: {state.first_speaker}")
    return state.to_dict()

# ============================================================================
# CLEANUP AND EXIT HANDLERS
# ============================================================================

def cleanup_interactive_session():
    """Clean up resources for interactive session"""
    global audio_streamer
    if audio_streamer:
        audio_streamer.stop()
    print("Interactive session cleaned up")

# Register cleanup
atexit.register(cleanup_interactive_session)

# ============================================================================
# ORIGINAL DEBATE NODES (FOR BACKWARD COMPATIBILITY)
# ============================================================================

def topic_setup_node(state: DebateState) -> Dict:
    print("\n=== TOPIC SETUP NODE ===")
    
    if not state.get("topic"):
        topic = "Should artificial intelligence be regulated by the government?"
        user_position = "AI should be regulated"
        ai_position = "AI should not be regulated"
    else:
        topic = state["topic"]
        prompt = f"""Given the debate topic: "{topic}"
        What are the two main opposing positions?
        Format: 
        FOR: [position supporting the proposition]
        AGAINST: [position opposing the proposition]"""
        
        response = llm.invoke([HumanMessage(content=prompt)])
        lines = response.content.strip().split('\n')
        user_position = lines[0].replace("FOR:", "").strip() if lines else "In favor"
        ai_position = lines[1].replace("AGAINST:", "").strip() if len(lines) > 1 else "Against"
    
    print(f"Topic: {topic}")
    print(f"User Position: {user_position}")
    print(f"AI Position: {ai_position}")
    
    debate_context = DebateContext(
        user_main_points=[], ai_main_points=[],
        unaddressed_points={"user": [], "ai": []},
        evidence_used={"user": [], "ai": []}
    )
    
    return {
        "topic": topic, "user_position": user_position, "ai_position": ai_position,
        "phase": DebatePhase.SETUP, "debate_active": True, "max_rounds": CONFIG["max_rounds"],
        "round_number": 1, "turn_count": 0, "debate_context": debate_context,
        "repetition_warnings": [], "argument_summaries": {}
    }

def voice_input_node(state: DebateState) -> Dict:
    print("\n=== VOICE INPUT NODE ===")
    
    # Stop any ongoing audio playback when it's user's turn
    audio_streamer.stop()
    
    print(f"Round {state['round_number']} - User's turn")
    print(f"Your position: {state['user_position']}")
    
    recent_args = [arg for arg in state.get("arguments", []) if arg["speaker"] == SpeakerType.AI]
    if recent_args and CONFIG["enable_context_awareness"]:
        print("\n--- Opponent's last argument ---")
        print(recent_args[-1]["content"][:300] + "...")
        print("--- End of opponent's argument ---\n")
    
    # Voice input logic
    user_argument = None
    if CONFIG["openai_api_key"]:
        use_voice = input("Use voice input? (y/n): ").lower() == 'y'
        if use_voice:
            print("\nSpeak clearly into your microphone...")
            input("Press Enter when ready to start listening...")
            user_argument = speech_to_text_whisper_api()
    
    # Fallback to text input
    if not user_argument:
        user_argument = input("\nEnter your argument: ")
    
    # Validation
    valid_input = False
    while not valid_input and user_argument:
        if len(user_argument) < CONFIG["min_argument_length"]:
            print(f"Argument too short. Minimum {CONFIG['min_argument_length']} characters.")
            user_argument = input("\nEnter your argument: ")
            continue
        
        is_repetitive, similar_to = detect_repetition(user_argument, state.get("arguments", []), SpeakerType.USER)
        if is_repetitive:
            print(f"\nWARNING: Similar to previous argument: '{similar_to}'")
            confirm = input("Submit anyway? (y/n): ")
            if confirm.lower() != 'y':
                user_argument = input("\nEnter your argument: ")
                continue
            else:
                state.get("repetition_warnings", []).append("User repeated argument")
        
        valid_input = True
    
    if not user_argument:
        user_argument = "I agree with my position and believe it's the right stance."
    
    # Process argument
    key_points = extract_key_points(user_argument, llm)
    rebuts_points = []
    if recent_args and CONFIG["enable_context_awareness"]:
        context = state.get("debate_context", {})
        ai_points = context.get("ai_main_points", [])
        for point in ai_points:
            if any(keyword in user_argument.lower() for keyword in point.lower().split()[:3]):
                rebuts_points.append(point)
    
    argument = Argument(
        speaker=SpeakerType.USER, content=user_argument, timestamp=time.time(),
        round_number=state["round_number"], key_points=key_points, rebuts_points=rebuts_points
    )
    
    context = state.get("debate_context", {})
    context["user_main_points"].extend(key_points)
    
    return {
        "current_argument": user_argument, "arguments": [argument],
        "turn_count": state.get("turn_count", 0) + 1, "debate_context": context
    }

# ============================================================================
# TEST FUNCTIONS
# ============================================================================

def test_murf_integration():
    """Test the streaming Murf integration."""
    print("\nTesting Streaming Murf AI integration...")
    
    if not MURF_CONFIG["api_key"]:
        print("MURF_API_KEY not set in environment variables")
        return False
    
    test_text = "Hello, this is a test of the Murf AI streaming system."
    print(f"Testing streaming with text: '{test_text}'")
    
    try:
        speak_text_streaming(test_text)
        print("Waiting for playback to complete...")
        audio_streamer.wait_until_complete(timeout=10)
        print("Streaming audio test successful!")
        return True
    except Exception as e:
        print(f"Streaming test failed: {e}")
        return False

# ============================================================================
# MAKE FUNCTIONS AVAILABLE FOR IMPORT
# ============================================================================

__all__ = [
    # Configuration exports
    'CONFIG', 'MURF_CONFIG', 'WHISPER_CONFIG',
    
    # Type exports
    'SpeakerType', 'DebatePhase', 'Argument', 'Score', 'DebateContext',
    'JudgeScore', 'ArgumentAnalysis',
    
    # Audio functions
    'speak_text_streaming', 'audio_streamer', 'speak_text_async',
    'speech_to_text_whisper_api', 'speech_to_text_whisper_streamlit',
    
    # API clients
    'openai_client',
    
    # Utility functions
    'extract_key_points', 'detect_repetition', 'analyze_opponent_argument',
    'check_similarity',
    
    # LLM functions
    'get_llm', 'llm', 'judge_llm',
    
    # Streamlit integration
    'InteractiveDebateManager', 'StreamlitDebateState',
    'get_interactive_manager', 'get_streamlit_state', 'get_current_state',
    'reset_debate_state', 'run_interactive_debate',
    'cleanup_interactive_session',
    
    # Test functions
    'test_murf_integration',
    
    # Legacy compatibility
    'StreamlitStateManager', 'state_manager',
    
    # Additional audio functions
    'speak_text', 'smart_split_text', 'text_to_speech_murf_corrected',
    'cleanup_audio'
]

print("Complete enhanced debate system with Streamlit integration loaded!")

# ============================================================================
# MAIN PROGRAM (TERMINAL MODE)
# ============================================================================

if __name__ == "__main__":
    print("ENHANCED DEBATE SYSTEM WITH STREAMING VOICE")
    print("=" * 60)
    print("Run 'streamlit run streamlit_app.py' for the interactive UI")
    print("=" * 60)
    # You can add your main logic here or remove the lines below if not needed.
    # print(f"Error: {e}")
    # return None

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def check_similarity(text1: str, text2: str) -> float:
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

def detect_repetition(new_argument: str, previous_arguments: List[Argument], speaker: str) -> Tuple[bool, Optional[str]]:
    if not CONFIG["enable_repetition_check"]:
        return False, None
    
    speaker_args = [arg for arg in previous_arguments if arg["speaker"] == speaker]
    for prev_arg in speaker_args:
        similarity = check_similarity(new_argument, prev_arg["content"])
        if similarity > CONFIG["similarity_threshold"]:
            return True, prev_arg["content"][:100] + "..."
    return False, None

def extract_key_points(argument: str, llm_instance) -> List[str]:
    prompt = f"""Extract 3-5 key points from this argument. Return only the key points as a bullet list.

Argument:
{argument}

Key points:"""
    
    response = llm_instance.invoke([HumanMessage(content=prompt)])
    points = response.content.strip().split('\n')
    return [p.strip('- •*').strip() for p in points if p.strip()]

def analyze_opponent_argument(argument: str, llm_instance) -> ArgumentAnalysis:
    parser = PydanticOutputParser(pydantic_object=ArgumentAnalysis)
    prompt = f"""Analyze this debate argument to identify main claims, evidence used, and weak points.

{parser.get_format_instructions()}

Argument to analyze:
{argument}"""
    
    response = llm_instance.invoke([HumanMessage(content=prompt)])
    try:
        return parser.parse(response.content)
    except:
        return ArgumentAnalysis(
            main_claims=["General argument provided"],
            evidence_used=[],
            weak_points=["Could use more specific evidence"]
        )

# Initialize LLM instances
llm = get_llm()
judge_llm = get_llm()
print(f"LLM initialized: {CONFIG['llm_type']}")

# ============================================================================
# STREAMLIT-COMPATIBLE VOICE FUNCTIONS
# ============================================================================

# ============================================================================
# STREAMLIT-COMPATIBLE VOICE FUNCTIONS
# ============================================================================

def speech_to_text_whisper_streamlit() -> Optional[str]:
    """Streamlit-compatible speech to text using Whisper API"""
    if not CONFIG["openai_api_key"]:
        return None
    
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    
    try:
        with microphone as source:
            # Adjust for ambient noise
            recognizer.adjust_for_ambient_noise(source, duration=1.0)
            
            # Set energy threshold
            recognizer.energy_threshold = WHISPER_CONFIG["energy_threshold"]
            
            # Listen with timeout
            audio = recognizer.listen(
                source, 
                timeout=WHISPER_CONFIG["timeout"],
                phrase_time_limit=WHISPER_CONFIG["phrase_time_limit"]
            )
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            with open(tmp.name, 'wb') as f:
                f.write(audio.get_wav_data())
            audio_path = tmp.name
        
        try:
            # Transcribe with Whisper
            with open(audio_path, 'rb') as audio_file:
                transcript = openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="en",
                    response_format="text"
                )
            
            # Clean up
            os.unlink(audio_path)
            
            if transcript and transcript.strip():
                return transcript.strip()
            return None
            
        except Exception as e:
            print(f"Transcription error: {e}")
            if os.path.exists(audio_path):
                os.unlink(audio_path)
            return None
            
    except sr.WaitTimeoutError:
        return None
    except Exception as e:
        print(f"Speech recognition error: {e}")
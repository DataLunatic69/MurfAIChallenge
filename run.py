# Cell 1: Enhanced Imports and Setup
import os
import json
import random
from typing import TypedDict, List, Dict, Literal, Optional, Annotated, Tuple
from enum import Enum
import operator
from datetime import datetime
import time
import re
from difflib import SequenceMatcher

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# LangChain imports
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

# For structured output
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

# For embeddings and similarity (preparation for better context)
import numpy as np
from typing import Deque
from collections import deque

print("Enhanced libraries imported successfully!")

import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv


load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

CONFIG = {
    "llm_type": "groq",
    "groq_model": "llama-3.1-70b-versatile",  # or "mixtral-8x7b-32768", "llama3-70b-8192"
    "groq_api_key": groq_api_key,
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



def get_llm():
    return ChatGroq(groq_api_key=groq_api_key, model_name="openai/gpt-oss-20b")

llm = get_llm()
judge_llm = get_llm()
print(f"LLM initialized: {CONFIG['llm_type']}")


# Cell 3: Structured Output Models for Better Parsing

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

class Score(TypedDict):
    user_score: float
    ai_score: float
    round_number: int
    reasoning: str

# Main State Definition
class DebateState(TypedDict):
    # Core debate information
    topic: str
    user_position: str  # Position user is arguing for
    ai_position: str    # Position AI is arguing for
    
    # Speaker management
    current_speaker: str
    first_speaker: Optional[str]
    
    # Arguments - using reducer to accumulate
    arguments: Annotated[List[Argument], operator.add]
    
    # Scoring - using reducer to accumulate
    scores: Annotated[List[Score], operator.add]
    
    # Debate flow control
    phase: str
    round_number: int
    max_rounds: int
    turn_count: int
    
    # Control flags
    debate_active: bool
    error_state: Optional[str]
    
    # Current processing
    current_argument: Optional[str]
    judge_feedback: Optional[str]

class JudgeScore(BaseModel):
    """Structured model for judge scoring."""
    logical_coherence: int = Field(
        ge=1, le=10,
        description="Score for logical coherence (1-10)"
    )
    evidence_support: int = Field(
        ge=1, le=10,
        description="Score for evidence and support (1-10)"
    )
    relevance: int = Field(
        ge=1, le=10,
        description="Score for relevance to topic (1-10)"
    )
    persuasiveness: int = Field(
        ge=1, le=10,
        description="Score for persuasiveness (1-10)"
    )
    total_score: int = Field(
        ge=4, le=40,
        description="Total score (sum of all criteria)"
    )
    reasoning: str = Field(
        description="Brief explanation of the scoring (2-3 sentences)"
    )
    strengths: List[str] = Field(
        default_factory=list,
        description="Key strengths of the argument"
    )
    weaknesses: List[str] = Field(
        default_factory=list,
        description="Areas for improvement"
    )

class ArgumentAnalysis(BaseModel):
    """Analysis of opponent's argument for better rebuttals."""
    main_claims: List[str] = Field(
        description="Main claims made by opponent"
    )
    evidence_used: List[str] = Field(
        description="Evidence or examples cited"
    )
    weak_points: List[str] = Field(
        description="Potential weaknesses or gaps"
    )
    
print("Structured output models defined!")

# Cell 4: Enhanced State with Context Tracking

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
    # New fields
    key_points: List[str]  # Extracted key points
    rebuts_points: List[str]  # Which opponent points it addresses

class Score(TypedDict):
    user_score: float
    ai_score: float
    round_number: int
    reasoning: str
    # New fields
    detailed_scores: Dict[str, int]  # Breakdown by criteria
    feedback: Dict[str, List[str]]  # Strengths and weaknesses

class DebateContext(TypedDict):
    """Track debate context for better responses."""
    user_main_points: List[str]
    ai_main_points: List[str]
    unaddressed_points: Dict[str, List[str]]  # Points not yet rebutted
    evidence_used: Dict[str, List[str]]  # Evidence by speaker
    
# Enhanced State Definition
class DebateState(TypedDict):
    # Core debate information
    topic: str
    user_position: str
    ai_position: str
    
    # Speaker management
    current_speaker: str
    first_speaker: Optional[str]
    
    # Arguments with metadata
    arguments: Annotated[List[Argument], operator.add]
    
    # Enhanced scoring
    scores: Annotated[List[Score], operator.add]
    
    # Debate flow control
    phase: str
    round_number: int
    max_rounds: int
    turn_count: int
    
    # Control flags
    debate_active: bool
    error_state: Optional[str]
    
    # Current processing
    current_argument: Optional[str]
    judge_feedback: Optional[str]
    
    # NEW: Context tracking
    debate_context: DebateContext
    repetition_warnings: Annotated[List[str], operator.add]
    argument_summaries: Dict[str, str]  # Round summaries

print("Enhanced state schema defined!")


# Cell 5: Utility Functions for Better Processing

def check_similarity(text1: str, text2: str) -> float:
    """Check similarity between two texts."""
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

def detect_repetition(new_argument: str, previous_arguments: List[Argument], 
                     speaker: str) -> Tuple[bool, Optional[str]]:
    """
    Detect if argument is repetitive.
    Returns (is_repetitive, similar_argument_excerpt)
    """
    if not CONFIG["enable_repetition_check"]:
        return False, None
    
    speaker_args = [arg for arg in previous_arguments if arg["speaker"] == speaker]
    
    for prev_arg in speaker_args:
        similarity = check_similarity(new_argument, prev_arg["content"])
        if similarity > CONFIG["similarity_threshold"]:
            return True, prev_arg["content"][:100] + "..."
    
    return False, None

def extract_key_points(argument: str, llm_instance) -> List[str]:
    """Extract key points from an argument."""
    prompt = f"""Extract 3-5 key points from this argument.
Return only the key points as a bullet list.

Argument:
{argument}

Key points:"""
    
    response = llm_instance.invoke([HumanMessage(content=prompt)])
    points = response.content.strip().split('\n')
    return [p.strip('- •*').strip() for p in points if p.strip()]

def analyze_opponent_argument(argument: str, llm_instance) -> ArgumentAnalysis:
    """Analyze opponent's argument for strategic response."""
    parser = PydanticOutputParser(pydantic_object=ArgumentAnalysis)
    
    prompt = f"""Analyze this debate argument to identify:
1. Main claims being made
2. Evidence or examples used
3. Potential weak points or gaps

{parser.get_format_instructions()}

Argument to analyze:
{argument}"""
    
    response = llm_instance.invoke([HumanMessage(content=prompt)])
    
    try:
        return parser.parse(response.content)
    except:
        # Fallback to basic analysis
        return ArgumentAnalysis(
            main_claims=["General argument provided"],
            evidence_used=[],
            weak_points=["Could use more specific evidence"]
        )

print("Utility functions loaded!")

# Cell 5: Utility Functions for Better Processing

def check_similarity(text1: str, text2: str) -> float:
    """Check similarity between two texts."""
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

def detect_repetition(new_argument: str, previous_arguments: List[Argument], 
                     speaker: str) -> Tuple[bool, Optional[str]]:
    """
    Detect if argument is repetitive.
    Returns (is_repetitive, similar_argument_excerpt)
    """
    if not CONFIG["enable_repetition_check"]:
        return False, None
    
    speaker_args = [arg for arg in previous_arguments if arg["speaker"] == speaker]
    
    for prev_arg in speaker_args:
        similarity = check_similarity(new_argument, prev_arg["content"])
        if similarity > CONFIG["similarity_threshold"]:
            return True, prev_arg["content"][:100] + "..."
    
    return False, None

def extract_key_points(argument: str, llm_instance) -> List[str]:
    """Extract key points from an argument."""
    prompt = f"""Extract 3-5 key points from this argument.
Return only the key points as a bullet list.

Argument:
{argument}

Key points:"""
    
    response = llm_instance.invoke([HumanMessage(content=prompt)])
    points = response.content.strip().split('\n')
    return [p.strip('- •*').strip() for p in points if p.strip()]

def analyze_opponent_argument(argument: str, llm_instance) -> ArgumentAnalysis:
    """Analyze opponent's argument for strategic response."""
    parser = PydanticOutputParser(pydantic_object=ArgumentAnalysis)
    
    prompt = f"""Analyze this debate argument to identify:
1. Main claims being made
2. Evidence or examples used
3. Potential weak points or gaps

{parser.get_format_instructions()}

Argument to analyze:
{argument}"""
    
    response = llm_instance.invoke([HumanMessage(content=prompt)])
    
    try:
        return parser.parse(response.content)
    except:
        # Fallback to basic analysis
        return ArgumentAnalysis(
            main_claims=["General argument provided"],
            evidence_used=[],
            weak_points=["Could use more specific evidence"]
        )

print("Utility functions loaded!")

# Cell 6: Enhanced Setup Nodes

def topic_setup_node(state: DebateState) -> Dict:
    """Initialize debate with better context setup."""
    print("\n=== ENHANCED TOPIC SETUP NODE ===")
    
    if not state.get("topic"):
        topic = "Should artificial intelligence be regulated by the government?"
        user_position = "AI should be regulated"
        ai_position = "AI should not be regulated"
    else:
        topic = state["topic"]
        # Better position extraction
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
    
    # Initialize context
    debate_context = DebateContext(
        user_main_points=[],
        ai_main_points=[],
        unaddressed_points={"user": [], "ai": []},
        evidence_used={"user": [], "ai": []}
    )
    
    return {
        "topic": topic,
        "user_position": user_position,
        "ai_position": ai_position,
        "phase": DebatePhase.SETUP,
        "debate_active": True,
        "max_rounds": CONFIG["max_rounds"],
        "round_number": 1,
        "turn_count": 0,
        "debate_context": debate_context,
        "repetition_warnings": [],
        "argument_summaries": {}
    }


# Cell 7: Enhanced User Input Node with Repetition Detection

def user_input_node(state: DebateState) -> Dict:
    """Enhanced user input with validation and repetition check."""
    print("\n=== ENHANCED USER INPUT NODE ===")
    print(f"Round {state['round_number']} - User's turn")
    print(f"Topic: {state['topic']}")
    print(f"Your position: {state['user_position']}")
    
    # Show opponent's last argument if exists
    recent_args = [arg for arg in state.get("arguments", []) 
                   if arg["speaker"] == SpeakerType.AI]
    if recent_args and CONFIG["enable_context_awareness"]:
        print("\n--- Opponent's last argument ---")
        print(recent_args[-1]["content"][:300] + "...")
        print("--- End of opponent's argument ---\n")
    
    # Get user input with validation
    valid_input = False
    while not valid_input:
        user_argument = input("\nEnter your argument: ")
        
        # Check minimum length
        if len(user_argument) < CONFIG["min_argument_length"]:
            print(f"⚠️ Argument too short. Please provide at least {CONFIG['min_argument_length']} characters.")
            continue
        
        # Check for repetition
        is_repetitive, similar_to = detect_repetition(
            user_argument, 
            state.get("arguments", []), 
            SpeakerType.USER
        )
        
        if is_repetitive:
            print(f"\n⚠️ WARNING: This argument is very similar to your previous argument:")
            print(f"'{similar_to}'")
            confirm = input("Do you want to submit this anyway? (y/n): ")
            if confirm.lower() != 'y':
                continue
            else:
                warning = f"User repeated argument from earlier round"
                state.get("repetition_warnings", []).append(warning)
        
        valid_input = True
    
    # Extract key points
    key_points = extract_key_points(user_argument, llm)
    
    # Analyze what opponent points are being addressed
    rebuts_points = []
    if recent_args and CONFIG["enable_context_awareness"]:
        context = state.get("debate_context", {})
        ai_points = context.get("ai_main_points", [])
        # Simple check - in production, use more sophisticated matching
        for point in ai_points:
            if any(keyword in user_argument.lower() for keyword in point.lower().split()[:3]):
                rebuts_points.append(point)
    
    # Create enhanced argument record
    argument = Argument(
        speaker=SpeakerType.USER,
        content=user_argument,
        timestamp=time.time(),
        round_number=state["round_number"],
        key_points=key_points,
        rebuts_points=rebuts_points
    )
    
    # Update context
    context = state.get("debate_context", {})
    context["user_main_points"].extend(key_points)
    
    return {
        "current_argument": user_argument,
        "arguments": [argument],
        "turn_count": state.get("turn_count", 0) + 1,
        "debate_context": context
    }

# Cell 8: Enhanced AI Debater with Context Awareness

def ai_debater_node(state: DebateState) -> Dict:
    """AI debater with better context awareness and rebuttal."""
    print("\n=== ENHANCED AI DEBATER NODE ===")
    print(f"Round {state['round_number']} - AI's turn")
    
    # Analyze opponent's last argument if exists
    recent_user_args = [arg for arg in state.get("arguments", []) 
                        if arg["speaker"] == SpeakerType.USER]
    
    opponent_analysis = None
    if recent_user_args and CONFIG["enable_context_awareness"]:
        last_user_arg = recent_user_args[-1]
        opponent_analysis = analyze_opponent_argument(last_user_arg["content"], llm)
    
    # Get previous AI arguments to avoid repetition
    ai_previous = [arg["content"] for arg in state.get("arguments", []) 
                   if arg["speaker"] == SpeakerType.AI]
    
    # Get judge feedback from last round
    recent_scores = state.get("scores", [])
    last_feedback = ""
    if recent_scores:
        for score in recent_scores[-2:]:  # Last 2 scores
            if score.get("ai_score", 0) > 0:
                feedback = score.get("feedback", {})
                weaknesses = feedback.get("weaknesses", [])
                if weaknesses:
                    last_feedback = f"Areas to improve: {', '.join(weaknesses)}"
    
    # Enhanced prompt with context
    system_prompt = f"""You are participating in a formal debate.
Topic: {state['topic']}
Your position: {state['ai_position']}
Current round: {state['round_number']} of {state['max_rounds']}

IMPORTANT GUIDELINES:
1. Make NEW arguments - do not repeat your previous points
2. Directly address and rebut opponent's specific claims
3. Use different evidence than before
4. Build upon feedback to strengthen your position
5. Be specific and concrete in your rebuttals

{f"Judge's feedback on your last argument: {last_feedback}" if last_feedback else ""}

You have already made these arguments - DO NOT REPEAT:
{chr(10).join([f"- {arg[:100]}..." for arg in ai_previous])}"""
    
    user_prompt_parts = []
    
    if opponent_analysis:
        user_prompt_parts.append(f"""Opponent's main claims to address:
{chr(10).join([f"- {claim}" for claim in opponent_analysis.main_claims])}

Weak points you can exploit:
{chr(10).join([f"- {weak}" for weak in opponent_analysis.weak_points])}""")
    
    user_prompt_parts.append(f"""
Round {state['round_number']} task:
{"Opening argument - establish your position strongly" if state['round_number'] == 1 
else "Rebuttal - counter opponent's claims while advancing new arguments" if state['round_number'] == 2
else "Closing - synthesize your strongest points and deliver final rebuttals"}

Provide your argument now:""")
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content="\n".join(user_prompt_parts))
    ]
    
    response = llm.invoke(messages)
    ai_argument = response.content
    
    # Check for self-repetition
    is_repetitive, _ = detect_repetition(
        ai_argument, 
        state.get("arguments", []), 
        SpeakerType.AI
    )
    
    if is_repetitive:
        # Try once more with stronger prompt
        messages.append(AIMessage(content=ai_argument))
        messages.append(HumanMessage(content="This argument is too similar to your previous ones. Please provide a completely different argument with new evidence and angles."))
        response = llm.invoke(messages)
        ai_argument = response.content
    
    print(f"\nAI Argument preview: {ai_argument[:200]}...")
    
    # Extract key points
    key_points = extract_key_points(ai_argument, llm)
    
    # Identify rebutted points
    rebuts_points = []
    if opponent_analysis:
        rebuts_points = opponent_analysis.main_claims[:2]  # Rebuts top claims
    
    # Create enhanced argument record
    argument = Argument(
        speaker=SpeakerType.AI,
        content=ai_argument,
        timestamp=time.time(),
        round_number=state["round_number"],
        key_points=key_points,
        rebuts_points=rebuts_points
    )
    
    # Update context
    context = state.get("debate_context", {})
    context["ai_main_points"].extend(key_points)
    
    return {
        "current_argument": ai_argument,
        "arguments": [argument],
        "turn_count": state.get("turn_count", 0) + 1,
        "debate_context": context
    }

# Cell 9: Enhanced Judge with Structured Scoring

def judge_node(state: DebateState) -> Dict:
    """Enhanced judge with structured output and detailed feedback."""
    print("\n=== ENHANCED JUDGE NODE ===")
    
    current_arg = state.get("current_argument", "")
    current_speaker = state.get("current_speaker", "")
    
    if not current_arg:
        print("No argument to judge")
        return {}
    
    # Check if argument addresses opponent's points
    current_arg_record = state.get("arguments", [])[-1] if state.get("arguments") else None
    rebuts_points = current_arg_record.get("rebuts_points", []) if current_arg_record else []
    
    # Create structured scoring prompt
    parser = PydanticOutputParser(pydantic_object=JudgeScore)
    
    system_prompt = f"""You are an impartial debate judge evaluating arguments.
Score each criterion from 1-10 and provide detailed feedback.

Scoring criteria:
- Logical coherence: Clear reasoning and structure
- Evidence/support: Use of facts, examples, data
- Relevance: Directly addresses the topic
- Persuasiveness: Compelling and convincing delivery

{f"Bonus consideration: This argument rebuts these opponent points: {rebuts_points}" if rebuts_points else ""}

{parser.get_format_instructions()}"""
    
    user_prompt = f"""Topic: {state['topic']}
Speaker: {current_speaker}
Position: {state['user_position'] if current_speaker == SpeakerType.USER else state['ai_position']}
Round: {state['round_number']} of {state['max_rounds']}

Argument to evaluate:
{current_arg}

Provide your structured evaluation:"""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    # Get structured response
    try:
        response = judge_llm.invoke(messages)
        evaluation = parser.parse(response.content)
    except Exception as e:
        print(f"Error parsing judge response: {e}")
        # Fallback scoring
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
    
    # Validate total score
    computed_total = (evaluation.logical_coherence + evaluation.evidence_support + 
                     evaluation.relevance + evaluation.persuasiveness)
    if abs(evaluation.total_score - computed_total) > 1:
        evaluation.total_score = computed_total
    
    print(f"Judge Score: {evaluation.total_score}/40")
    print(f"Breakdown: Logic={evaluation.logical_coherence}, Evidence={evaluation.evidence_support}, "
          f"Relevance={evaluation.relevance}, Persuasive={evaluation.persuasiveness}")
    print(f"Reasoning: {evaluation.reasoning[:150]}...")
    
    # Create enhanced score record
    score = Score(
        user_score=evaluation.total_score if current_speaker == SpeakerType.USER else 0,
        ai_score=evaluation.total_score if current_speaker == SpeakerType.AI else 0,
        round_number=state["round_number"],
        reasoning=evaluation.reasoning,
        detailed_scores={
            "logical_coherence": evaluation.logical_coherence,
            "evidence_support": evaluation.evidence_support,
            "relevance": evaluation.relevance,
            "persuasiveness": evaluation.persuasiveness
        },
        feedback={
            "strengths": evaluation.strengths,
            "weaknesses": evaluation.weaknesses
        }
    )
    
    return {
        "scores": [score],
        "judge_feedback": evaluation.reasoning
    }


# Cell 10: Enhanced Moderator with Better Round Management

# Cell 2: Fixed Moderator Node
def moderator_node(state: DebateState) -> Dict:
    """Enhanced moderator with context tracking - FIXED turn management."""
    print("\n=== ENHANCED MODERATOR NODE ===")
    
    current_round = state.get("round_number", 1)
    max_rounds = state.get("max_rounds", CONFIG["max_rounds"])
    turn_count = state.get("turn_count", 0)
    
    print(f"Current Round: {current_round}/{max_rounds}")
    print(f"Turn Count: {turn_count}")
    
    # Check for repetition warnings
    warnings = state.get("repetition_warnings", [])
    if warnings:
        print(f"⚠️ Repetition warnings: {len(warnings)}")
    
    # Check if debate should end
    if current_round > max_rounds:
        print("Debate complete - max rounds reached")
        return {
            "debate_active": False,
            "phase": DebatePhase.CONCLUSION,
            "current_speaker": SpeakerType.MODERATOR,
            "argument_summaries": state.get("argument_summaries", {})
        }
    
    # FIXED: Determine next speaker based on turn count and who has spoken
    current_speaker = state.get("current_speaker", "")
    
    # Get speakers who have already gone in this round
    speakers_in_round = [
        arg["speaker"] for arg in state.get("arguments", [])
        if arg["round_number"] == current_round
    ]
    
    # FIXED LOGIC: 
    # If turn_count is 0 (start of debate), use the first_speaker
    # Otherwise, alternate based on who just went
    if turn_count == 0:
        # First turn of the debate - use the selected first speaker
        next_speaker = state.get("first_speaker", SpeakerType.USER)
    else:
        # After someone has spoken, determine who goes next
        if current_speaker == SpeakerType.USER:
            next_speaker = SpeakerType.AI
        elif current_speaker == SpeakerType.AI:
            next_speaker = SpeakerType.USER
        else:
            # Fallback: if current_speaker is not set properly
            # Check who hasn't gone this round
            if SpeakerType.USER not in speakers_in_round:
                next_speaker = SpeakerType.USER
            elif SpeakerType.AI not in speakers_in_round:
                next_speaker = SpeakerType.AI
            else:
                # Both have spoken, move to next round
                next_speaker = state.get("first_speaker", SpeakerType.USER)
    
    # Check if we need to move to next round
    both_spoke = (
        SpeakerType.USER in speakers_in_round and
        SpeakerType.AI in speakers_in_round
    )
    
    new_round = current_round
    if both_spoke:
        new_round = current_round + 1
        print(f"Moving to round {new_round}")
        
        # Generate round summary
        round_summary = f"Round {current_round}: "
        for arg in state.get("arguments", []):
            if arg["round_number"] == current_round:
                key_points = arg.get("key_points", [])
                round_summary += f"{arg['speaker']}: {', '.join(key_points[:2])}. "
        
        summaries = state.get("argument_summaries", {})
        summaries[f"round_{current_round}"] = round_summary
        print(f"Round {current_round} Summary: {round_summary}")
        
        # For new round, alternate who starts (optional - you can keep same order)
        # next_speaker = SpeakerType.AI if state.get("first_speaker") == SpeakerType.USER else SpeakerType.USER
    
    if turn_count != 0:
        print(f"Next speaker: {next_speaker}")
    
    return {
        "current_speaker": next_speaker,
        "round_number": new_round,
        "debate_active": new_round <= max_rounds,
        "argument_summaries": summaries if 'summaries' in locals() else state.get("argument_summaries", {})
    }

# Cell 11: Enhanced Final Judgment with Detailed Analysis

# Cell 2: Complete Final Judgment Node

def final_judgment_node(state: DebateState) -> Dict:
    """Enhanced final judgment with comprehensive analysis."""
    print("\n=== ENHANCED FINAL JUDGMENT NODE ===")
    
    # Calculate detailed scores
    all_scores = state.get("scores", [])
    
    total_user_score = sum(s.get("user_score", 0) for s in all_scores)
    total_ai_score = sum(s.get("ai_score", 0) for s in all_scores)
    
    # Calculate average scores by criteria
    user_criteria_scores = {"logical_coherence": [], "evidence_support": [], 
                           "relevance": [], "persuasiveness": []}
    ai_criteria_scores = {"logical_coherence": [], "evidence_support": [], 
                         "relevance": [], "persuasiveness": []}
    
    for score in all_scores:
        detailed = score.get("detailed_scores", {})
        if score.get("user_score", 0) > 0:
            for criterion, value in detailed.items():
                user_criteria_scores[criterion].append(value)
        elif score.get("ai_score", 0) > 0:
            for criterion, value in detailed.items():
                ai_criteria_scores[criterion].append(value)
    
    # Calculate averages
    user_avg_criteria = {k: np.mean(v) if v else 0 for k, v in user_criteria_scores.items()}
    ai_avg_criteria = {k: np.mean(v) if v else 0 for k, v in ai_criteria_scores.items()}
    
    # Determine winner
    if total_user_score > total_ai_score:
        winner = "USER"
        margin = total_user_score - total_ai_score
    elif total_ai_score > total_user_score:
        winner = "AI"
        margin = total_ai_score - total_user_score
    else:
        winner = "TIE"
        margin = 0
    
    # Analyze debate quality
    arguments = state.get("arguments", [])
    context = state.get("debate_context", {})
    warnings = state.get("repetition_warnings", [])
    
    # Count rebuttals
    total_rebuttals = sum(len(arg.get("rebuts_points", [])) for arg in arguments)
    
    # Count unique evidence points
    user_evidence = len(context.get('evidence_used', {}).get('user', []))
    ai_evidence = len(context.get('evidence_used', {}).get('ai', []))
    total_evidence = user_evidence + ai_evidence
    
    # Generate comprehensive summary
    summary = f"""
╔════════════════════════════════════════════════════════════╗
║                    DEBATE FINAL JUDGMENT                   ║
╚════════════════════════════════════════════════════════════╝

TOPIC: {state['topic']}
FORMAT: {state.get('round_number', 0) - 1} rounds completed

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FINAL SCORES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
User Total: {total_user_score:.1f}/120
AI Total:   {total_ai_score:.1f}/120

Winner: {winner} {"(by " + str(margin) + " points)" if winner != "TIE" else ""}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DETAILED CRITERIA ANALYSIS (Averages)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    USER        AI
Logical Coherence:  {user_avg_criteria['logical_coherence']:.1f}/10      {ai_avg_criteria['logical_coherence']:.1f}/10
Evidence Support:   {user_avg_criteria['evidence_support']:.1f}/10      {ai_avg_criteria['evidence_support']:.1f}/10
Relevance:          {user_avg_criteria['relevance']:.1f}/10      {ai_avg_criteria['relevance']:.1f}/10
Persuasiveness:     {user_avg_criteria['persuasiveness']:.1f}/10      {ai_avg_criteria['persuasiveness']:.1f}/10

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DEBATE QUALITY METRICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Arguments: {len(arguments)}
Total Rebuttals: {total_rebuttals}
Repetition Warnings: {len(warnings)}
Unique Evidence Points: {total_evidence}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
KEY ARGUMENTS BY ROUND
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""
    
    # Add round summaries
    summaries = state.get("argument_summaries", {})
    for round_key, round_summary in summaries.items():
        summary += f"\n{round_summary}"
    
    # Add judge's key feedback
    summary += f"""

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
JUDGE'S OBSERVATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""
    
    # Compile strengths and weaknesses
    user_strengths = []
    user_weaknesses = []
    ai_strengths = []
    ai_weaknesses = []
    
    for score in all_scores:
        feedback = score.get("feedback", {})
        if score.get("user_score", 0) > 0:
            user_strengths.extend(feedback.get("strengths", []))
            user_weaknesses.extend(feedback.get("weaknesses", []))
        elif score.get("ai_score", 0) > 0:
            ai_strengths.extend(feedback.get("strengths", []))
            ai_weaknesses.extend(feedback.get("weaknesses", []))
    
    # Remove duplicates and get top items
    user_strengths = list(set(user_strengths))[:3]
    user_weaknesses = list(set(user_weaknesses))[:3]
    ai_strengths = list(set(ai_strengths))[:3]
    ai_weaknesses = list(set(ai_weaknesses))[:3]
    
    summary += f"""

User Strengths:
{chr(10).join([f"  • {s}" for s in user_strengths]) if user_strengths else "  • No specific strengths noted"}

User Areas for Improvement:
{chr(10).join([f"  • {w}" for w in user_weaknesses]) if user_weaknesses else "  • No specific weaknesses noted"}

AI Strengths:
{chr(10).join([f"  • {s}" for s in ai_strengths]) if ai_strengths else "  • No specific strengths noted"}

AI Areas for Improvement:
{chr(10).join([f"  • {w}" for w in ai_weaknesses]) if ai_weaknesses else "  • No specific weaknesses noted"}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FINAL VERDICT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""
    
    # Generate final verdict explanation
    if winner == "USER":
        verdict = f"The USER wins this debate with a {margin}-point margin. "
        if user_avg_criteria['evidence_support'] > ai_avg_criteria['evidence_support']:
            verdict += "The user provided stronger evidence and support for their arguments. "
        if user_avg_criteria['logical_coherence'] > ai_avg_criteria['logical_coherence']:
            verdict += "The user's arguments were more logically coherent. "
        if user_avg_criteria['persuasiveness'] > ai_avg_criteria['persuasiveness']:
            verdict += "The user presented more persuasive arguments overall. "
        if user_avg_criteria['relevance'] > ai_avg_criteria['relevance']:
            verdict += "The user maintained better relevance to the topic throughout. "
    elif winner == "AI":
        verdict = f"The AI wins this debate with a {margin}-point margin. "
        if ai_avg_criteria['evidence_support'] > user_avg_criteria['evidence_support']:
            verdict += "The AI provided stronger evidence and support for its arguments. "
        if ai_avg_criteria['logical_coherence'] > user_avg_criteria['logical_coherence']:
            verdict += "The AI's arguments were more logically coherent. "
        if ai_avg_criteria['persuasiveness'] > user_avg_criteria['persuasiveness']:
            verdict += "The AI presented more persuasive arguments overall. "
        if ai_avg_criteria['relevance'] > user_avg_criteria['relevance']:
            verdict += "The AI maintained better relevance to the topic throughout. "
    else:
        verdict = "This debate ends in a tie. Both participants presented equally strong arguments. "
        verdict += "The debate showcased excellent argumentation from both sides with comparable "
        verdict += "logical coherence, evidence support, relevance, and persuasiveness."
    
    summary += f"\n{verdict}"
    
    # Add debate highlights
    summary += f"""

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DEBATE HIGHLIGHTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🏆 Best Scoring Round: {"User" if max(user_criteria_scores.get('logical_coherence', [0])) > max(ai_criteria_scores.get('logical_coherence', [0])) else "AI"} 
📊 Most Evidence Used: {"User" if user_evidence > ai_evidence else "AI" if ai_evidence > user_evidence else "Tie"}
🎯 Most Rebuttals Made: {"User" if sum(1 for arg in arguments if arg["speaker"] == "user" and arg.get("rebuts_points")) > sum(1 for arg in arguments if arg["speaker"] == "ai" and arg.get("rebuts_points")) else "AI"}
💡 Strongest Criterion: 
   - User: {max(user_avg_criteria, key=user_avg_criteria.get) if user_avg_criteria else "N/A"}
   - AI: {max(ai_avg_criteria, key=ai_avg_criteria.get) if ai_avg_criteria else "N/A"}

═══════════════════════════════════════════════════════════════
                    END OF DEBATE JUDGMENT
═══════════════════════════════════════════════════════════════"""
    
    print(summary)
    
    return {
        "phase": DebatePhase.COMPLETE,
        "debate_active": False,
        "final_summary": summary,
        "final_scores": {
            "user_total": total_user_score,
            "ai_total": total_ai_score,
            "winner": winner,
            "margin": margin,
            "user_criteria_avg": user_avg_criteria,
            "ai_criteria_avg": ai_avg_criteria
        }
    }

def speaker_selection_node(state: DebateState) -> Dict:
    """
    Determine who speaks first in the debate.
    """
    print("\n=== SPEAKER SELECTION NODE ===")
    
    # Random selection for Phase 1
    first_speaker = random.choice([SpeakerType.USER, SpeakerType.AI])
    
    print(f"First speaker selected: {first_speaker}")
    
    return {
        "first_speaker": first_speaker,
        "current_speaker": first_speaker,
        "phase": DebatePhase.DEBATE
    }

def route_from_moderator(state: DebateState) -> Literal["user_input", "ai_debater", "final_judgment"]:
    """
    Route from moderator based on debate state.
    """
    if not state.get("debate_active", True):
        return "final_judgment"
    
    next_speaker = state.get("current_speaker", "")
    
    if next_speaker == SpeakerType.USER:
        return "user_input"
    elif next_speaker == SpeakerType.AI:
        return "ai_debater"
    else:
        return "final_judgment"

def route_after_judge(state: DebateState) -> Literal["moderator", "final_judgment"]:
    """
    Route after judge evaluation.
    """
    if state.get("debate_active", True):
        return "moderator"
    else:
        return "final_judgment"


# Cell 12: Updated Graph Builder with Enhanced Features

def build_enhanced_debate_graph():
    """Build the enhanced debate graph with all improvements."""
    print("Building enhanced debate graph...")
    
    # Initialize the graph
    workflow = StateGraph(DebateState)
    
    # Add all nodes
    workflow.add_node("topic_setup", topic_setup_node)
    workflow.add_node("speaker_selection", speaker_selection_node)
    workflow.add_node("user_input", user_input_node)
    workflow.add_node("ai_debater", ai_debater_node)
    workflow.add_node("judge", judge_node)
    workflow.add_node("moderator", moderator_node)
    workflow.add_node("final_judgment", final_judgment_node)
    
    # Add edges - Setup flow
    workflow.add_edge(START, "topic_setup")
    workflow.add_edge("topic_setup", "speaker_selection")
    workflow.add_edge("speaker_selection", "moderator")
    
    # From moderator, route to appropriate speaker or end
    workflow.add_conditional_edges(
        "moderator",
        route_from_moderator,
        {
            "user_input": "user_input",
            "ai_debater": "ai_debater",
            "final_judgment": "final_judgment"
        }
    )
    
    # From speakers to judge
    workflow.add_edge("user_input", "judge")
    workflow.add_edge("ai_debater", "judge")
    
    # From judge, either back to moderator or to final judgment
    workflow.add_conditional_edges(
        "judge",
        route_after_judge,
        {
            "moderator": "moderator",
            "final_judgment": "final_judgment"
        }
    )
    
    # Final judgment to END
    workflow.add_edge("final_judgment", END)
    
    # Compile with checkpointer
    checkpointer = MemorySaver()
    app = workflow.compile(checkpointer=checkpointer)
    
    print("Enhanced graph built successfully!")
    return app

# Build the enhanced graph
enhanced_debate_app = build_enhanced_debate_graph()
# Cell 3: Basic State Definition

# Cell 13: Enhanced Testing Utilities

def run_enhanced_debate(topic: str = None, config: dict = None):
    """Run an enhanced debate session with all improvements."""
    print("=" * 60)
    print("STARTING ENHANCED DEBATE SESSION")
    print("=" * 60)
    print("\nEnhancements Active:")
    print(f"✓ Repetition Detection: {CONFIG['enable_repetition_check']}")
    print(f"✓ Context Awareness: {CONFIG['enable_context_awareness']}")
    print(f"✓ Structured Scoring: Enabled")
    print(f"✓ Rebuttal Tracking: Enabled")
    print(f"✓ Round Summaries: Enabled")
    print("=" * 60)
    
    # Initial state with enhanced fields
    initial_state = {
        "topic": topic,
        "arguments": [],
        "scores": [],
        "debate_active": True,
        "round_number": 1,
        "turn_count": 0,
        "debate_context": DebateContext(
            user_main_points=[],
            ai_main_points=[],
            unaddressed_points={"user": [], "ai": []},
            evidence_used={"user": [], "ai": []}
        ),
        "repetition_warnings": [],
        "argument_summaries": {}
    }
    
    # Configuration for the run
    run_config = {
        "configurable": {
            "thread_id": f"enhanced_debate_{datetime.now().timestamp()}"
        }
    }
    
    if config:
        run_config.update(config)
    
    try:
        # Run the debate
        final_state = enhanced_debate_app.invoke(initial_state, run_config)
        
        print("\n" + "=" * 60)
        print("ENHANCED DEBATE COMPLETED")
        print("=" * 60)
        
        return final_state
    
    except Exception as e:
        print(f"Error during debate: {e}")
        import traceback
        traceback.print_exc()
        return None

print("State schema defined successfully!")



def run_enhanced_debate(topic: str = None, config: dict = None):
    """Run an enhanced debate session with all improvements."""
    print("=" * 60)
    print("STARTING ENHANCED DEBATE SESSION")
    print("=" * 60)
    print("\nEnhancements Active:")
    print(f"✓ Repetition Detection: {CONFIG['enable_repetition_check']}")
    print(f"✓ Context Awareness: {CONFIG['enable_context_awareness']}")
    print(f"✓ Structured Scoring: Enabled")
    print(f"✓ Rebuttal Tracking: Enabled")
    print(f"✓ Round Summaries: Enabled")
    print("=" * 60)
    
    # Initial state with enhanced fields
    initial_state = {
        "topic": topic,
        "arguments": [],
        "scores": [],
        "debate_active": True,
        "round_number": 1,
        "turn_count": 0,
        "debate_context": DebateContext(
            user_main_points=[],
            ai_main_points=[],
            unaddressed_points={"user": [], "ai": []},
            evidence_used={"user": [], "ai": []}
        ),
        "repetition_warnings": [],
        "argument_summaries": {}
    }
    
    # Configuration for the run
    run_config = {
        "configurable": {
            "thread_id": f"enhanced_debate_{datetime.now().timestamp()}"
        }
    }
    
    if config:
        run_config.update(config)
    
    try:
        # Run the debate
        final_state = enhanced_debate_app.invoke(initial_state, run_config)
        
        print("\n" + "=" * 60)
        print("ENHANCED DEBATE COMPLETED")
        print("=" * 60)
        
        return final_state
    
    except Exception as e:
        print(f"Error during debate: {e}")
        import traceback
        traceback.print_exc()
        return None

def display_enhanced_transcript(state: dict):
    """Display an enhanced debate transcript with all metadata."""
    print("\n" + "=" * 60)
    print("ENHANCED DEBATE TRANSCRIPT")
    print("=" * 60)
    
    print(f"\n📋 Topic: {state.get('topic', 'No topic')}")
    print(f"👤 User Position: {state.get('user_position', '')}")
    print(f"🤖 AI Position: {state.get('ai_position', '')}")
    print("\n" + "-" * 60)
    
    arguments = state.get("arguments", [])
    scores = state.get("scores", [])
    
    for i, arg in enumerate(arguments):
        # Find corresponding score
        round_scores = [s for s in scores if s["round_number"] == arg["round_number"]]
        arg_score = None
        for score in round_scores:
            if (arg["speaker"] == SpeakerType.USER and score.get("user_score", 0) > 0) or \
               (arg["speaker"] == SpeakerType.AI and score.get("ai_score", 0) > 0):
                arg_score = score
                break
        
        print(f"\n{'='*60}")
        print(f"Round {arg['round_number']} - {arg['speaker'].upper()}")
        print(f"{'='*60}")
        
        # Display key points
        if arg.get("key_points"):
            print("\n📌 Key Points:")
            for point in arg["key_points"][:3]:
                print(f"  • {point}")
        
        # Display rebuttals
        if arg.get("rebuts_points"):
            print("\n🎯 Addresses opponent's points:")
            for point in arg["rebuts_points"][:2]:
                print(f"  • {point}")
        
        # Display argument (truncated)
        print(f"\n💬 Argument:")
        print(arg['content'][:400])
        if len(arg['content']) > 400:
            print("...")
        
        # Display score if available
        if arg_score:
            detailed = arg_score.get("detailed_scores", {})
            total = arg_score.get("user_score", 0) or arg_score.get("ai_score", 0)
            print(f"\n⚖️ Judge Score: {total}/40")
            if detailed:
                print(f"  Logic: {detailed.get('logical_coherence', 0)}/10 | "
                      f"Evidence: {detailed.get('evidence_support', 0)}/10 | "
                      f"Relevance: {detailed.get('relevance', 0)}/10 | "
                      f"Persuasive: {detailed.get('persuasiveness', 0)}/10")
            
            feedback = arg_score.get("feedback", {})
            if feedback.get("strengths"):
                print(f"  ✓ Strengths: {', '.join(feedback['strengths'][:2])}")
            if feedback.get("weaknesses"):
                print(f"  ✗ Areas to improve: {', '.join(feedback['weaknesses'][:2])}")
    
    # Display warnings if any
    warnings = state.get("repetition_warnings", [])
    if warnings:
        print("\n" + "=" * 60)
        print("⚠️ WARNINGS")
        print("=" * 60)
        for warning in warnings:
            print(f"  • {warning}")
    
    # Display final summary if available
    if state.get("final_summary"):
        print("\n" + state["final_summary"])


test_topics = [
    "Should social media platforms be held responsible for user-generated content?",
    "Is artificial general intelligence an existential threat to humanity?",
    "Should governments implement universal basic income?",
    "Is remote work better than office work for productivity?",
]

print("🎯 ENHANCED DEBATE SYSTEM READY")
print("=" * 60)
print("Choose a topic for the enhanced debate:")
for i, topic in enumerate(test_topics, 1):
    print(f"{i}. {topic}")
print("5. Custom topic")

choice = input("\nSelect (1-5): ")

if choice == "5":
    debate_topic = input("Enter your custom topic: ")
elif choice in ["1", "2", "3", "4"]:
    debate_topic = test_topics[int(choice) - 1]
else:
    debate_topic = test_topics[0]

print(f"\n📋 Selected Topic: {debate_topic}")
print("\n⚡ Enhanced Features Active:")
print("• Real-time repetition detection")
print("• Context-aware rebuttals")
print("• Structured scoring (4 criteria)")
print("• Round summaries")
print("• Detailed final analysis")
print("\nPress Enter to begin the enhanced debate...")
input()

# Run the enhanced debate
final_state = run_enhanced_debate(topic=debate_topic)

# Display the enhanced transcript
if final_state:
    display_enhanced_transcript(final_state)
    
    # Save transcript to file
    save_option = input("\n💾 Save transcript to file? (y/n): ")
    if save_option.lower() == 'y':
        filename = f"debate_transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            # Convert non-serializable items
            save_state = {
                k: v for k, v in final_state.items() 
                if k not in ['debate_context']  # Skip complex objects
            }
            json.dump(save_state, f, indent=2, default=str)
        print(f"✅ Transcript saved to {filename}")
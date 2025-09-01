# streamlit_app.py - Professional Debate Chamber UI
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

# Professional CSS with React-like styling
def inject_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    :root {
        --primary: #3b82f6;
        --primary-hover: #2563eb;
        --primary-light: #60a5fa;
        --secondary: #64748b;
        --accent: #06b6d4;
        --accent-hover: #0891b2;
        --success: #10b981;
        --success-hover: #059669;
        --warning: #f59e0b;
        --warning-hover: #d97706;
        --danger: #ef4444;
        --danger-hover: #dc2626;
        --dark: #0f172a;
        --dark-light: #1e293b;
        --dark-medium: #334155;
        --dark-lighter: #475569;
        --light: #f8fafc;
        --light-gray: #e2e8f0;
        --border: #e2e8f0;
        --border-dark: rgba(255, 255, 255, 0.1);
        --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
        --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
        --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
        --shadow-xl: 0 20px 25px -5px rgb(0 0 0 / 0.1), 0 8px 10px -6px rgb(0 0 0 / 0.1);
        --shadow-2xl: 0 25px 50px -12px rgb(0 0 0 / 0.25);
        --border-radius: 0.75rem;
        --border-radius-lg: 1rem;
        --border-radius-xl: 1.5rem;
        --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        --transition-fast: all 0.15s cubic-bezier(0.4, 0, 0.2, 1);
        --gradient-primary: linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%);
        --gradient-success: linear-gradient(135deg, var(--success) 0%, #34d399 100%);
        --gradient-warning: linear-gradient(135deg, var(--warning) 0%, #fbbf24 100%);
        --gradient-danger: linear-gradient(135deg, var(--danger) 0%, #f87171 100%);
    }
    
    /* Reset and Base Styles */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 25%, #312e81 50%, #1e1b4b 75%, #0f172a 100%);
        background-size: 400% 400%;
        animation: gradientShift 20s ease infinite;
        color: #ffffff;
        line-height: 1.6;
        min-height: 100vh;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .main .block-container {
        padding: 2rem 1rem;
        max-width: 1400px;
        margin: 0 auto;
    }
    
    /* Enhanced Header Components */
    .debate-masthead {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.15) 0%, rgba(6, 182, 212, 0.1) 50%, rgba(59, 130, 246, 0.05) 100%);
        backdrop-filter: blur(30px);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: var(--border-radius-xl);
        padding: 4rem 2rem;
        margin-bottom: 3rem;
        text-align: center;
        position: relative;
        overflow: hidden;
        box-shadow: var(--shadow-2xl);
    }
    
    .debate-masthead::before {
        content: '';
        position: absolute;
        top: 0;
        left: -50%;
        width: 200%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.05), transparent);
        animation: shimmer 4s infinite;
    }
    
    .debate-masthead::after {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(59, 130, 246, 0.1) 0%, transparent 70%);
        animation: rotate 10s linear infinite;
    }
    
    .debate-masthead h1 {
        font-size: 4rem;
        font-weight: 900;
        margin: 0 0 1.5rem 0;
        background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 50%, #ffffff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.03em;
        position: relative;
        z-index: 2;
        text-shadow: 0 0 30px rgba(255, 255, 255, 0.3);
    }
    
    .debate-masthead .subtitle {
        font-size: 1.375rem;
        font-weight: 400;
        color: #94a3b8;
        margin: 0;
        position: relative;
        z-index: 2;
        opacity: 0.9;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    @keyframes rotate {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Enhanced Card Components */
    .card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(25px);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: var(--border-radius-lg);
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: var(--shadow-xl);
        transition: var(--transition);
        position: relative;
        overflow: hidden;
    }
    
    .card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(135deg, #3b82f6 0%, #06b6d4 100%);
        opacity: 0;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .card:hover {
        border-color: rgba(59, 130, 246, 0.4);
        box-shadow: 0 25px 50px -12px rgb(0 0 0 / 0.25);
        transform: translateY(-4px);
    }
    
    .card:hover::before {
        opacity: 1;
    }
    
    .card-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 1.5rem;
        padding-bottom: 1.5rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .card-title {
        font-size: 1.375rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 0.875rem;
    }
    
    .card-badge {
        background: linear-gradient(135deg, #3b82f6 0%, #06b6d4 100%);
        color: white;
        padding: 0.375rem 1rem;
        border-radius: 9999px;
        font-size: 0.875rem;
        font-weight: 600;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Enhanced Argument Cards */
    .argument-card {
        background: rgba(255, 255, 255, 0.04);
        border: 2px solid transparent;
        border-radius: var(--border-radius-lg);
        padding: 2.5rem;
        margin: 2rem 0;
        transition: var(--transition);
        position: relative;
        overflow: hidden;
        box-shadow: var(--shadow-lg);
    }
    
    .argument-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(135deg, #3b82f6 0%, #06b6d4 100%);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .argument-card::after {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255, 255, 255, 0.02) 0%, transparent 70%);
        opacity: 0;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .argument-card:hover::after {
        opacity: 1;
    }
    
    .user-argument {
        border-color: rgba(59, 130, 246, 0.4);
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.08) 0%, rgba(6, 182, 212, 0.04) 100%);
    }
    
    .user-argument::before {
        background: linear-gradient(135deg, #3b82f6 0%, #06b6d4 100%);
    }
    
    .ai-argument {
        border-color: rgba(6, 182, 212, 0.4);
        background: linear-gradient(135deg, rgba(6, 182, 212, 0.08) 0%, rgba(14, 165, 233, 0.04) 100%);
    }
    
    .ai-argument::before {
        background: linear-gradient(135deg, #3b82f6 0%, #06b6d4 100%);
    }
    
    .judge-card {
        border-color: rgba(245, 158, 11, 0.4);
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.08) 0%, rgba(251, 191, 36, 0.04) 100%);
    }
    
    .judge-card::before {
        background: linear-gradient(135deg, #f59e0b 0%, #fbbf24 100%);
    }
    
    .argument-content {
        font-size: 1.125rem;
        line-height: 1.75;
        color: #e2e8f0;
        margin: 1.5rem 0;
        position: relative;
        z-index: 2;
    }
    
    .argument-meta {
        display: flex;
        align-items: center;
        gap: 1.5rem;
        margin-top: 2rem;
        padding-top: 1.5rem;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        font-size: 0.875rem;
        color: #94a3b8;
        position: relative;
        z-index: 2;
    }
    
    /* Enhanced Status Components */
    .status-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 1.5rem;
        margin: 2.5rem 0;
    }
    
    .status-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: var(--border-radius-lg);
        padding: 2rem;
        text-align: center;
        transition: var(--transition);
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(20px);
    }
    
    .status-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(135deg, #3b82f6 0%, #06b6d4 100%);
        opacity: 0;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .status-card:hover {
        border-color: rgba(59, 130, 246, 0.4);
        transform: translateY(-3px);
        box-shadow: 0 20px 25px -5px rgb(0 0 0 / 0.1), 0 8px 10px -6px rgb(0 0 0 / 0.1);
    }
    
    .status-card:hover::before {
        opacity: 1;
    }
    
    .status-label {
        font-size: 0.875rem;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.75rem;
    }
    
    .status-value {
        font-size: 2rem;
        font-weight: 800;
        color: #ffffff;
        text-shadow: 0 0 20px rgba(255, 255, 255, 0.3);
    }
    
    .status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.75rem 1.5rem;
        border-radius: 9999px;
        font-size: 0.875rem;
        font-weight: 600;
        margin: 0.5rem;
        box-shadow: var(--shadow-md);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .status-active {
        background: linear-gradient(135deg, #3b82f6 0%, #06b6d4 100%);
        color: white;
    }
    
    .status-waiting {
        background: linear-gradient(135deg, #f59e0b 0%, #fbbf24 100%);
        color: white;
    }
    
    .status-complete {
        background: linear-gradient(135deg, #3b82f6 0%, #06b6d4 100%);
        color: white;
    }
    
    /* Enhanced Turn Indicators */
    .turn-banner {
        background: linear-gradient(135deg, #3b82f6 0%, #06b6d4 100%);
        color: white;
        padding: 3rem 2rem;
        border-radius: 1.5rem;
        text-align: center;
        margin: 3rem 0;
        box-shadow: 0 25px 50px -12px rgb(0 0 0 / 0.25);
        animation: pulseGlow 3s infinite alternate;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .turn-banner::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255, 255, 255, 0.15) 0%, transparent 70%);
        animation: rotate 6s linear infinite;
    }
    
    .turn-banner h2 {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0 0 1rem 0;
        position: relative;
        z-index: 2;
        text-shadow: 0 0 30px rgba(255, 255, 255, 0.5);
    }
    
    .turn-banner p {
        font-size: 1.25rem;
        margin: 0;
        opacity: 0.95;
        position: relative;
        z-index: 2;
        font-weight: 500;
    }
    
    @keyframes pulseGlow {
        0% { 
            box-shadow: var(--shadow-2xl);
            transform: scale(1);
        }
        100% { 
            box-shadow: 0 30px 60px rgba(59, 130, 246, 0.6);
            transform: scale(1.02);
        }
    }
    
    /* Enhanced Score Display */
    .score-container {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 2.5rem;
        margin: 3rem 0;
    }
    
    .score-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(25px);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: var(--border-radius-xl);
        padding: 3rem 2rem;
        text-align: center;
        transition: var(--transition);
        position: relative;
        overflow: hidden;
        box-shadow: var(--shadow-xl);
    }
    
    .score-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(135deg, #3b82f6 0%, #06b6d4 100%);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .score-card:hover {
        transform: translateY(-6px);
        box-shadow: 0 25px 50px -12px rgb(0 0 0 / 0.25);
        border-color: rgba(59, 130, 246, 0.4);
    }
    
    .score-card:hover::before {
        height: 6px;
    }
    
    .score-label {
        font-size: 1.25rem;
        font-weight: 700;
        color: #94a3b8;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.75rem;
    }
    
    .score-value {
        font-size: 5rem;
        font-weight: 900;
        color: #ffffff;
        margin: 1.5rem 0;
        line-height: 1;
        text-shadow: 0 0 40px rgba(255, 255, 255, 0.4);
        background: linear-gradient(135deg, #3b82f6 0%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .score-max {
        font-size: 1.125rem;
        color: #64748b;
        font-weight: 600;
    }
    
    /* Enhanced Input Components */
    .input-section {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(25px);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: var(--border-radius-xl);
        padding: 3rem;
        margin: 3rem 0;
        box-shadow: var(--shadow-xl);
        position: relative;
        overflow: hidden;
    }
    
    .input-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(135deg, #3b82f6 0%, #06b6d4 100%);
    }
    
    .input-header {
        margin-bottom: 2.5rem;
    }
    
    .input-title {
        font-size: 1.75rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.75rem;
    }
    
    .input-subtitle {
        color: #94a3b8;
        font-size: 1.125rem;
        font-weight: 500;
    }
    
    .voice-button {
        background: linear-gradient(135deg, #3b82f6 0%, #06b6d4 100%);
        border: none;
        color: white;
        padding: 1.25rem 2.5rem;
        border-radius: 1rem;
        font-size: 1.125rem;
        font-weight: 700;
        cursor: pointer;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 1rem;
        width: 100%;
        min-height: 70px;
        box-shadow: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        text-transform: none;
        letter-spacing: 0;
    }
    
    .voice-button:hover {
        transform: translateY(-3px);
        box-shadow: 0 25px 50px -12px rgb(0 0 0 / 0.25);
        background: linear-gradient(135deg, #2563eb 0%, #0891b2 100%);
    }
    
    .voice-button:active {
        transform: translateY(-1px);
    }
    
    .listening-pulse {
        animation: pulse 1.5s infinite;
        box-shadow: 0 0 30px rgba(59, 130, 246, 0.6);
    }
    
    @keyframes pulse {
        0%, 100% { 
            opacity: 1;
            transform: scale(1);
        }
        50% { 
            opacity: 0.8;
            transform: scale(1.05);
        }
    }
    
    /* Enhanced Button Components */
    .btn-primary {
        background: linear-gradient(135deg, #3b82f6 0%, #06b6d4 100%);
        border: none;
        color: white;
        padding: 1rem 2.5rem;
        border-radius: 1rem;
        font-size: 1.125rem;
        font-weight: 700;
        cursor: pointer;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        text-transform: none;
        letter-spacing: 0;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 0.75rem;
        min-height: 56px;
        box-shadow: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .btn-primary:hover {
        transform: translateY(-3px);
        box-shadow: 0 25px 50px -12px rgb(0 0 0 / 0.25);
        background: linear-gradient(135deg, #2563eb 0%, #0891b2 100%);
    }
    
    .btn-secondary {
        background: rgba(100, 116, 139, 0.15);
        border: 1px solid rgba(100, 116, 139, 0.4);
        color: #94a3b8;
        padding: 1rem 2.5rem;
        border-radius: 1rem;
        font-size: 1.125rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        backdrop-filter: blur(20px);
    }
    
    .btn-secondary:hover {
        background: rgba(100, 116, 139, 0.25);
        color: #ffffff;
        transform: translateY(-2px);
        border-color: rgba(100, 116, 139, 0.6);
    }
    
    /* Enhanced Sidebar Styling */
    .css-1d391kg {
        background: rgba(15, 23, 42, 0.95);
        backdrop-filter: blur(30px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .sidebar-content {
        padding: 1.5rem;
    }
    
    .sidebar-section {
        margin-bottom: 2.5rem;
        padding-bottom: 2rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .sidebar-section:last-child {
        border-bottom: none;
    }
    
    .sidebar-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    /* Enhanced Connection Status */
    .connection-status {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.75rem;
        margin-top: 1.5rem;
    }
    
    .connection-item {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.75rem;
        border-radius: var(--border-radius);
        font-size: 0.875rem;
        font-weight: 600;
        box-shadow: var(--shadow-sm);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .connection-success {
        background: linear-gradient(135deg, #10b981 0%, #34d399 100%);
        color: white;
    }
    
    .connection-error {
        background: linear-gradient(135deg, #ef4444 0%, #f87171 100%);
        color: white;
    }
    
    /* Enhanced Loading States */
    .loading-overlay {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(15, 23, 42, 0.9);
        backdrop-filter: blur(12px);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 1000;
    }
    
    .loading-spinner {
        width: 80px;
        height: 80px;
        border: 6px solid rgba(255, 255, 255, 0.1);
        border-top: 6px solid var(--primary);
        border-radius: 50%;
        animation: spin 1s linear infinite;
        box-shadow: 0 0 30px rgba(59, 130, 246, 0.3);
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Enhanced Form Controls */
    .stTextArea textarea,
    .stTextInput input,
    .stSelectbox select {
        background: rgba(255, 255, 255, 0.06) !important;
        border: 2px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: var(--border-radius-lg) !important;
        color: #ffffff !important;
        font-size: 1rem !important;
        padding: 1rem !important;
        transition: var(--transition) !important;
        backdrop-filter: blur(20px) !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    .stTextArea textarea:focus,
    .stTextInput input:focus,
    .stSelectbox select:focus {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.15) !important;
        outline: none !important;
        background: rgba(255, 255, 255, 0.08) !important;
    }
    
    .stSlider {
        padding: 1.5rem 0 !important;
    }
    
    .stCheckbox label {
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
    }
    
    /* Enhanced Animations */
    .fade-in {
        animation: fadeIn 0.8s ease-out;
    }
    
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .slide-in-left {
        animation: slideInLeft 0.6s ease-out;
    }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-40px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    .slide-in-right {
        animation: slideInRight 0.6s ease-out;
    }
    
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(40px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    .bounce-in {
        animation: bounceIn 0.8s ease-out;
    }
    
    @keyframes bounceIn {
        0% {
            opacity: 0;
            transform: scale(0.3);
        }
        50% {
            opacity: 1;
            transform: scale(1.05);
        }
        70% {
            transform: scale(0.9);
        }
        100% {
            opacity: 1;
            transform: scale(1);
        }
    }
    
    /* Enhanced Responsive Design */
    @media (max-width: 768px) {
        .debate-masthead {
            padding: 2.5rem 1rem;
        }
        
        .debate-masthead h1 {
            font-size: 2.75rem;
        }
        
        .score-container {
            grid-template-columns: 1fr;
            gap: 1.5rem;
        }
        
        .status-grid {
            grid-template-columns: 1fr;
        }
        
        .argument-card {
            padding: 1.75rem;
        }
        
        .input-section {
            padding: 2rem;
        }
        
        .turn-banner {
            padding: 2rem 1rem;
        }
        
        .turn-banner h2 {
            font-size: 2rem;
        }
    }
    
    /* Enhanced Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 12px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 6px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, var(--primary) 0%, var(--accent) 100%);
        border-radius: 6px;
        border: 2px solid rgba(255, 255, 255, 0.1);
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, var(--primary-hover) 0%, var(--accent-hover) 100%);
    }
    
    /* Enhanced Overrides for Streamlit */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #06b6d4 100%) !important;
        border: none !important;
        border-radius: 1rem !important;
        color: white !important;
        font-weight: 700 !important;
        padding: 1rem 2.5rem !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        text-transform: none !important;
        letter-spacing: 0 !important;
        min-height: 56px !important;
        box-shadow: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        font-size: 1.125rem !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 25px 50px -12px rgb(0 0 0 / 0.25) !important;
        background: linear-gradient(135deg, #2563eb 0%, #0891b2 100%) !important;
    }
    
    .stInfo,
    .stSuccess,
    .stWarning,
    .stError {
        border-radius: 1rem !important;
        border: none !important;
        backdrop-filter: blur(20px) !important;
        padding: 1.5rem !important;
        box-shadow: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1) !important;
    }
    
    .stInfo {
        background: rgba(59, 130, 246, 0.15) !important;
        color: #93c5fd !important;
        border: 1px solid rgba(59, 130, 246, 0.3) !important;
    }
    
    .stSuccess {
        background: rgba(16, 185, 129, 0.15) !important;
        color: #6ee7b7 !important;
        border: 1px solid rgba(16, 185, 129, 0.3) !important;
    }
    
    .stWarning {
        background: rgba(245, 158, 11, 0.15) !important;
        color: #fbbf24 !important;
        border: 1px solid rgba(245, 158, 11, 0.3) !important;
    }
    
    .stError {
        background: rgba(239, 68, 68, 0.15) !important;
        color: #fca5a5 !important;
        border: 1px solid rgba(239, 68, 68, 0.3) !important;
    }
    
    /* Additional Enhancements */
    .glow-effect {
        box-shadow: 0 0 30px rgba(59, 130, 246, 0.3);
    }
    
    .text-gradient {
        background: linear-gradient(135deg, #3b82f6 0%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .glass-effect {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
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

# Main debate controller (unchanged logic, using original class)
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

# Professional UI Components
def render_header():
    st.markdown("""
    <div class="debate-masthead fade-in">
        <h1>⚖️ Murf Debate Chamber</h1>
        <p class="subtitle">Professional debate platform with real-time AI interaction and expert judging</p>
        <div style="margin-top: 2rem; display: flex; justify-content: center; gap: 1.5rem; flex-wrap: wrap;">
            <div style="display: flex; align-items: center; gap: 0.75rem; padding: 0.75rem 1.5rem; background: rgba(59, 130, 246, 0.15); border: 1px solid rgba(59, 130, 246, 0.3); border-radius: 50px; backdrop-filter: blur(10px); box-shadow: 0 4px 15px rgba(59, 130, 246, 0.2);">
                <span style="font-size: 1.25rem;">🎯</span>
                <span style="font-weight: 600; color: #60a5fa;">Real-time AI</span>
            </div>
            <div style="display: flex; align-items: center; gap: 0.75rem; padding: 0.75rem 1.5rem; background: rgba(6, 182, 212, 0.15); border: 1px solid rgba(6, 182, 212, 0.3); border-radius: 50px; backdrop-filter: blur(10px); box-shadow: 0 4px 15px rgba(6, 182, 212, 0.2);">
                <span style="font-size: 1.25rem;">⚖️</span>
                <span style="font-weight: 600; color: #22d3ee;">Expert Judging</span>
            </div>
            <div style="display: flex; align-items: center; gap: 0.75rem; padding: 0.75rem 1.5rem; background: rgba(245, 158, 11, 0.15); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 50px; backdrop-filter: blur(10px); box-shadow: 0 4px 15px rgba(245, 158, 11, 0.2);">
                <span style="font-size: 1.25rem;">🎙️</span>
                <span style="font-weight: 600; color: #fbbf24;">Voice Input</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-content">
            <div class="sidebar-section">
                <h3 class="sidebar-title">🎯 Debate Configuration</h3>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        topics = [
            "Should AI be regulated by the government?",
            "Is remote work better than office work?",
            "Should social media be held responsible for content?",
            "Is UBI necessary for the future?",
            "Custom topic"
        ]
        
        selected = st.selectbox("Select Topic", topics, key="topic_select")
        
        if selected == "Custom topic":
            custom = st.text_input("Enter your topic:", key="custom_topic")
            topic = custom if custom else topics[0]
        else:
            topic = selected
        
        st.session_state.max_rounds = st.slider("Number of Rounds", 1, 5, 3, key="rounds_slider")
        
        st.markdown("""
        <div class="sidebar-section">
            <h3 class="sidebar-title">🎙️ Voice Settings</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.session_state.voice_enabled = st.checkbox("Enable Voice Input", value=True, key="voice_check")
        st.session_state.auto_advance = st.checkbox("Auto-advance Rounds", value=True, key="auto_check")
        
        st.markdown("""
        <div class="sidebar-section">
            <h3 class="sidebar-title">📡 Connection Status</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Connection status
        st.markdown('<div class="connection-status">', unsafe_allow_html=True)
        
        if CONFIG.get("openai_api_key"):
            st.markdown('<div class="connection-item connection-success">✓ OpenAI</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="connection-item connection-error">✗ OpenAI</div>', unsafe_allow_html=True)
            
        if MURF_CONFIG.get("api_key"):
            st.markdown('<div class="connection-item connection-success">✓ Murf AI</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="connection-item connection-error">✗ Murf AI</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        return topic

def render_debate_status():
    """Show current debate status with professional styling"""
    st.markdown('<div class="status-grid">', unsafe_allow_html=True)
    
    # Round indicator with progress
    progress_percent = (st.session_state.current_round / st.session_state.max_rounds) * 100
    st.markdown(f"""
    <div class="status-card">
        <div class="status-label">🎯 Round Progress</div>
        <div class="status-value">{st.session_state.current_round} / {st.session_state.max_rounds}</div>
        <div style="margin-top: 1rem; background: rgba(255,255,255,0.1); height: 6px; border-radius: 3px; overflow: hidden;">
            <div style="width: {progress_percent}%; height: 100%; background: linear-gradient(135deg, #3b82f6 0%, #06b6d4 100%); transition: width 0.5s ease;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Phase indicator with icon
    phase_icon = "🎬" if st.session_state.debate_phase == 'setup' else "⚔️" if st.session_state.debate_phase == 'debate' else "🏆"
    phase_text = st.session_state.debate_phase.title()
    st.markdown(f"""
    <div class="status-card">
        <div class="status-label">{phase_icon} Phase</div>
        <div class="status-value">{phase_text}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Current speaker with active indicator
    if st.session_state.current_speaker:
        speaker = "You" if st.session_state.current_speaker == 'user' else "AI"
        speaker_icon = "👤" if st.session_state.current_speaker == 'user' else "🤖"
        active_class = "status-active" if st.session_state.debate_active else "status-waiting"
        st.markdown(f"""
        <div class="status-card">
            <div class="status-label">🎭 Current Turn</div>
            <div class="status-value">{speaker_icon} {speaker}</div>
            <div class="status-indicator {active_class}" style="margin-top: 1rem; font-size: 0.75rem;">
                <span>{'🟢 Active' if st.session_state.debate_active else '🟡 Waiting'}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Arguments count with trend
    total_args = len(st.session_state.arguments)
    args_icon = "💬"
    st.markdown(f"""
    <div class="status-card">
        <div class="status-label">{args_icon} Arguments</div>
        <div class="status-value">{total_args}</div>
        <div style="margin-top: 0.5rem; font-size: 0.875rem; color: #94a3b8;">
            {total_args // 2 if total_args > 0 else 0} exchanges
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_score_display():
    """Display current scores with professional styling"""
    total_user = sum(s['user_score'] for s in st.session_state.scores)
    total_ai = sum(s['ai_score'] for s in st.session_state.scores)
    max_possible = st.session_state.max_rounds * 40
    
    # Calculate percentages for progress bars
    user_percent = (total_user / max_possible) * 100 if max_possible > 0 else 0
    ai_percent = (total_ai / max_possible) * 100 if max_possible > 0 else 0
    
    # Determine winner for visual emphasis
    winner = "user" if total_user > total_ai else "ai" if total_ai > total_user else "tie"
    
    st.markdown('<div class="score-container">', unsafe_allow_html=True)
    
    # User score with enhanced styling
    user_class = "score-card slide-in-left"
    if winner == "user":
        user_class += " glow-effect"
    
    st.markdown(f"""
    <div class="{user_class}">
        <div class="score-label">
            <span>👤 Your Score</span>
            {'🏆' if winner == 'user' else ''}
        </div>
        <div class="score-value">{total_user}</div>
        <div class="score-max">/ {max_possible}</div>
        <div style="margin-top: 1.5rem; background: rgba(255,255,255,0.1); height: 8px; border-radius: 4px; overflow: hidden;">
            <div style="width: {user_percent}%; height: 100%; background: linear-gradient(135deg, #3b82f6 0%, #06b6d4 100%); transition: width 0.8s ease; box-shadow: 0 0 10px rgba(59, 130, 246, 0.5);"></div>
        </div>
        <div style="margin-top: 0.5rem; font-size: 0.875rem; color: #94a3b8;">
            {user_percent:.1f}% of max score
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # AI score with enhanced styling
    ai_class = "score-card slide-in-right"
    if winner == "ai":
        ai_class += " glow-effect"
    
    st.markdown(f"""
    <div class="{ai_class}">
        <div class="score-label">
            <span>🤖 AI Score</span>
            {'🏆' if winner == 'ai' else ''}
        </div>
        <div class="score-value">{total_ai}</div>
        <div class="score-max">/ {max_possible}</div>
        <div style="margin-top: 1.5rem; background: rgba(255,255,255,0.1); height: 8px; border-radius: 4px; overflow: hidden;">
            <div style="width: {ai_percent}%; height: 100%; background: linear-gradient(135deg, #3b82f6 0%, #06b6d4 100%); transition: width 0.8s ease; box-shadow: 0 0 10px rgba(59, 130, 246, 0.5);"></div>
        </div>
        <div style="margin-top: 0.5rem; font-size: 0.875rem; color: #94a3b8;">
            {ai_percent:.1f}% of max score
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Score difference indicator
    if total_user != total_ai:
        diff = abs(total_user - total_ai)
        leader = "You" if total_user > total_ai else "AI"
        st.markdown(f"""
        <div style="text-align: center; margin: 2rem 0; padding: 1rem; background: rgba(255,255,255,0.03); border-radius: var(--border-radius-lg); border: 1px solid rgba(255,255,255,0.1);">
            <div style="font-size: 1.125rem; font-weight: 600; color: #ffffff;">
                🏁 {leader} leads by {diff} points
            </div>
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
    """Render user input area with professional styling"""
    if st.session_state.current_speaker == 'user' and st.session_state.debate_active:
        # Turn announcement banner
        st.markdown(f"""
        <div class="turn-banner">
            <h2>🎯 Your Turn to Speak!</h2>
            <p>Present your argument for: <strong>{st.session_state.user_position}</strong></p>
            <p>Round {st.session_state.current_round} of {st.session_state.max_rounds}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Stop any audio playback
        audio_streamer.stop()
    
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="input-header">
            <h3 class="input-title">Make Your Argument</h3>
            <p class="input-subtitle">Choose your preferred input method below</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if st.session_state.voice_enabled and openai_client:
                voice_class = "voice-button listening-pulse" if st.session_state.is_listening else "voice-button"
                voice_text = "🎤 Listening..." if st.session_state.is_listening else "🎤 Voice Input"
                voice_icon = "🔴" if st.session_state.is_listening else "🎤"
                
                st.markdown(f"""
                <div style="margin-bottom: 1rem;">
                    <div class="{voice_class}" onclick="document.getElementById('voice_btn').click()">
                        <span>{voice_icon}</span>
                        <span>{voice_text}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("Voice Input", key="voice_btn", help="Click to start voice input"):
                    with st.spinner("🎤 Listening for your argument..."):
                        text = capture_voice_input()
                        if text:
                            st.session_state.user_argument = text
                            st.rerun()

        with col2:
            st.markdown("""
            <div style="margin-bottom: 1rem;">
                <label style="font-weight: 600; color: #ffffff; margin-bottom: 0.5rem; display: block;">
                    ✍️ Type your argument:
                </label>
            </div>
            """, unsafe_allow_html=True)
            
            text_input = st.text_area(
                "",
                value=st.session_state.user_argument,
                height=120,
                key="text_input",
                placeholder="Enter your compelling argument here... Make it count! 💪"
            )
            
            if text_input:
                st.session_state.user_argument = text_input
        
        if st.session_state.user_argument:
            char_count = len(st.session_state.user_argument)
            word_count = len(st.session_state.user_argument.split())
            
            st.markdown(f"""
            <div class="card bounce-in">
                <div class="card-header">
                    <h4 class="card-title">📝 Your Argument Preview</h4>
                    <div class="card-badge">{word_count} words</div>
                </div>
                <div class="argument-content">
                    {st.session_state.user_argument[:300]}{'...' if len(st.session_state.user_argument) > 300 else ''}
                </div>
                <div class="argument-meta">
                    <span>📊 {char_count} characters</span>
                    <span>•</span>
                    <span>📝 {word_count} words</span>
                    <span>•</span>
                    <span>⏱️ {word_count // 3} sec read</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("Submit Argument", type="primary", use_container_width=True, key="submit_btn"):
                success, msg = controller.process_user_argument(st.session_state.user_argument)
                if success:
                    # Judge the argument
                    with st.spinner("⚖️ Judge is evaluating your argument..."):
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
                    st.session_state.turn_announced = False
                    
                    # Check if round is complete after both speakers
                    if controller.check_round_complete():
                        controller.advance_round()
                    
                    st.rerun()
                else:
                    st.error(msg)
        
        st.markdown('</div>', unsafe_allow_html=True)

def render_ai_turn(controller):
    """Handle AI's turn with professional presentation"""
    if st.session_state.current_speaker == 'ai' and st.session_state.debate_active:
        with st.spinner("🤖 AI is formulating response..."):
            time.sleep(2)  # Dramatic pause
            
            # Generate AI argument
            ai_argument = controller.generate_ai_argument()
            
            # Display AI argument
            st.markdown(f"""
            <div class="argument-card ai-argument fade-in">
                <div class="card-header">
                    <h4 class="card-title">🤖 AI Debater</h4>
                    <div class="card-badge">Round {st.session_state.current_round}</div>
                </div>
                <div class="argument-content">{ai_argument}</div>
                <div class="argument-meta">
                    <span>Position: {st.session_state.ai_position}</span>
                    <span>•</span>
                    <span>{datetime.now().strftime('%H:%M:%S')}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Speak AI argument
            if MURF_CONFIG["api_key"]:
                with st.spinner("🔊 AI is speaking..."):
                    speak_text_streaming(ai_argument)
                    audio_streamer.wait_until_complete()
            
            # Judge the argument
            with st.spinner("⚖️ Judge is evaluating AI's argument..."):
                time.sleep(1)
                evaluation = controller.judge_argument(ai_argument, 'ai')
                
                # Store judge evaluation for transcript
                st.session_state.last_ai_evaluation = {
                    'reasoning': evaluation.reasoning,
                    'score': evaluation.total_score,
                    'round': st.session_state.current_round,
                    'timestamp': time.time()
                }
                
                # Display judge feedback
                st.markdown(f"""
                <div class="argument-card judge-card fade-in">
                    <div class="card-header">
                        <h4 class="card-title">⚖️ Judge's Evaluation</h4>
                        <div class="card-badge">{evaluation.total_score}/40</div>
                    </div>
                    <div class="argument-content">{evaluation.reasoning}</div>
                    <div class="argument-meta">
                        <span>Logic: {evaluation.logical_coherence}/10</span>
                        <span>•</span>
                        <span>Evidence: {evaluation.evidence_support}/10</span>
                        <span>•</span>
                        <span>Relevance: {evaluation.relevance}/10</span>
                        <span>•</span>
                        <span>Persuasion: {evaluation.persuasiveness}/10</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                if MURF_CONFIG["api_key"]:
                    speak_text_streaming(f"Judge feedback: {evaluation.reasoning}")
                    audio_streamer.wait_until_complete()
            
            # Switch back to user's turn
            st.session_state.current_speaker = 'user'
            st.session_state.turn_announced = False
            
            # Check if round is complete after both speakers
            if controller.check_round_complete():
                controller.advance_round()
            
            # Auto-advance if enabled
            if st.session_state.auto_advance:
                time.sleep(2)
                st.rerun()

def render_debate_transcript():
    """Display debate transcript with professional styling"""
    if st.session_state.arguments:
        st.markdown("""
        <div class="card">
            <div class="card-header">
                <h3 class="card-title">📄 Debate Transcript</h3>
                <div class="card-badge">{} Arguments</div>
            </div>
        </div>
        """.format(len(st.session_state.arguments)), unsafe_allow_html=True)
        
        for i, arg in enumerate(st.session_state.arguments):
            card_class = "user-argument" if arg['speaker'] == 'user' else "ai-argument"
            speaker_icon = "👤" if arg['speaker'] == 'user' else "🤖"
            speaker_name = "You" if arg['speaker'] == 'user' else "AI Debater"
            
            st.markdown(f"""
            <div class="argument-card {card_class} fade-in">
                <div class="card-header">
                    <h4 class="card-title">{speaker_icon} {speaker_name}</h4>
                    <div class="card-badge">Round {arg['round_number']}</div>
                </div>
                <div class="argument-content">{arg['content']}</div>
                <div class="argument-meta">
                    <span>{datetime.fromtimestamp(arg['timestamp']).strftime('%H:%M:%S')}</span>
                </div>
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
                <div class="argument-card judge-card">
                    <div class="card-header">
                        <h4 class="card-title">⚖️ Judge's Feedback</h4>
                        <div class="card-badge">{score}/40</div>
                    </div>
                    <div class="argument-content">{relevant_score['reasoning']}</div>
                </div>
                """, unsafe_allow_html=True)

def render_final_results():
    """Display final debate results with celebration"""
    if st.session_state.debate_phase == 'complete':
        total_user = sum(s['user_score'] for s in st.session_state.scores)
        total_ai = sum(s['ai_score'] for s in st.session_state.scores)
        
        # Winner announcement with enhanced styling
        if "User wins" in st.session_state.final_winner:
            winner_emoji = "🎉"
            winner_class = "turn-banner"
            winner_gradient = "linear-gradient(135deg, #3b82f6 0%, #06b6d4 100%)"
        elif "AI wins" in st.session_state.final_winner:
            winner_emoji = "🤖"
            winner_class = "turn-banner"
            winner_gradient = "linear-gradient(135deg, #3b82f6 0%, #06b6d4 100%)"
        else:
            winner_emoji = "🤝"
            winner_class = "turn-banner"
            winner_gradient = "linear-gradient(135deg, #f59e0b 0%, #fbbf24 100%)"
        
        st.markdown(f"""
        <div class="{winner_class}" style="background: {winner_gradient};">
            <h2>{winner_emoji} Debate Complete!</h2>
            <p><strong>{st.session_state.final_winner}</strong></p>
            <p>Final Scores - You: {total_user} | AI: {total_ai}</p>
            <div style="margin-top: 2rem; display: flex; justify-content: center; gap: 1.5rem; flex-wrap: wrap;">
                <div style="display: flex; align-items: center; gap: 0.75rem; padding: 0.75rem 1.5rem; background: rgba(59, 130, 246, 0.15); border: 1px solid rgba(59, 130, 246, 0.3); border-radius: 50px; backdrop-filter: blur(10px); box-shadow: 0 4px 15px rgba(59, 130, 246, 0.2);">
                    <span style="font-size: 1.25rem;">🏆</span>
                    <span style="font-weight: 600; color: #60a5fa;">Champion</span>
                </div>
                <div style="display: flex; align-items: center; gap: 0.75rem; padding: 0.75rem 1.5rem; background: rgba(6, 182, 212, 0.15); border: 1px solid rgba(6, 182, 212, 0.3); border-radius: 50px; backdrop-filter: blur(10px); box-shadow: 0 4px 15px rgba(6, 182, 212, 0.2);">
                    <span style="font-size: 1.25rem;">📊</span>
                    <span style="font-weight: 600; color: #22d3ee;">Results</span>
                </div>
                <div style="display: flex; align-items: center; gap: 0.75rem; padding: 0.75rem 1.5rem; background: rgba(245, 158, 11, 0.15); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 50px; backdrop-filter: blur(10px); box-shadow: 0 4px 15px rgba(245, 158, 11, 0.2);">
                    <span style="font-size: 1.25rem;">🎯</span>
                    <span style="font-weight: 600; color: #fbbf24;">Complete</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Detailed breakdown
        st.markdown("""
        <div class="card">
            <div class="card-header">
                <h3 class="card-title">📊 Detailed Results</h3>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Round-by-round breakdown
        for i in range(1, st.session_state.max_rounds + 1):
            round_scores = [s for s in st.session_state.scores if s['round_number'] == i]
            if round_scores:
                user_round_score = sum(s['user_score'] for s in round_scores)
                ai_round_score = sum(s['ai_score'] for s in round_scores)
                
                st.markdown(f"""
                <div class="status-card">
                    <div class="status-label">Round {i}</div>
                    <div class="status-value">You: {user_round_score} | AI: {ai_round_score}</div>
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
            st.markdown(f"""
            <div class="card bounce-in">
                <div class="card-header">
                    <h3 class="card-title">🎯 Selected Topic</h3>
                    <div class="card-badge">Ready</div>
                </div>
                <div class="argument-content" style="font-size: 1.25rem; font-weight: 600; color: #ffffff; text-align: center; padding: 2rem; background: rgba(59, 130, 246, 0.1); border-radius: var(--border-radius); border: 1px solid rgba(59, 130, 246, 0.2);">
                    "{topic}"
                </div>
                <div style="margin-top: 2rem; text-align: center;">
                    <div style="display: flex; justify-content: center; gap: 1.5rem; margin-bottom: 1rem; flex-wrap: wrap;">
                        <div style="display: flex; align-items: center; gap: 0.75rem; padding: 0.75rem 1.5rem; background: rgba(59, 130, 246, 0.15); border: 1px solid rgba(59, 130, 246, 0.3); border-radius: 50px; backdrop-filter: blur(10px); box-shadow: 0 4px 15px rgba(59, 130, 246, 0.2);">
                            <span style="font-size: 1.25rem;">⚖️</span>
                            <span style="font-weight: 600; color: #60a5fa;">AI Judge</span>
                        </div>
                        <div style="display: flex; align-items: center; gap: 0.75rem; padding: 0.75rem 1.5rem; background: rgba(6, 182, 212, 0.15); border: 1px solid rgba(6, 182, 212, 0.3); border-radius: 50px; backdrop-filter: blur(10px); box-shadow: 0 4px 15px rgba(6, 182, 212, 0.2);">
                            <span style="font-size: 1.25rem;">🎙️</span>
                            <span style="font-weight: 600; color: #22d3ee;">Voice Input</span>
                        </div>
                        <div style="display: flex; align-items: center; gap: 0.75rem; padding: 0.75rem 1.5rem; background: rgba(245, 158, 11, 0.15); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 50px; backdrop-filter: blur(10px); box-shadow: 0 4px 15px rgba(245, 158, 11, 0.2);">
                            <span style="font-size: 1.25rem;">📊</span>
                            <span style="font-weight: 600; color: #fbbf24;">Real-time Scoring</span>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div style="text-align: center; margin: 2rem 0;">
                <div style="font-size: 1.125rem; color: #94a3b8; margin-bottom: 1rem;">
                    Ready to engage in intellectual combat? 🚀
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🚀 Start Debate", type="primary", use_container_width=True, key="start_debate"):
                with st.spinner("⚙️ Setting up debate chamber..."):
                    time.sleep(1)  # Dramatic pause
                    controller.setup_debate(topic)
                st.rerun()
    
    # Active debate area
    if st.session_state.debate_active:
        render_debate_status()
        render_score_display()
        
        # Show positions with enhanced styling
        st.markdown(f"""
        <div class="card fade-in">
            <div class="card-header">
                <h3 class="card-title">⚖️ Debate Positions</h3>
                <div class="card-badge">Active</div>
            </div>
            <div class="argument-content">
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; margin: 1rem 0;">
            <div style="padding: 1.5rem; background: rgba(59, 130, 246, 0.1); border-radius: var(--border-radius); border: 1px solid rgba(59, 130, 246, 0.2);">
                <div style="font-weight: 700; color: #60a5fa; margin-bottom: 0.5rem;">👤 Your Position</div>
                <div style="color: #e2e8f0;">{st.session_state.user_position}</div>
            </div>
            <div style="padding: 1.5rem; background: rgba(6, 182, 212, 0.1); border-radius: var(--border-radius); border: 1px solid rgba(6, 182, 212, 0.2);">
                <div style="font-weight: 700; color: #22d3ee; margin-bottom: 0.5rem;">🤖 AI Position</div>
                <div style="color: #e2e8f0;">{st.session_state.ai_position}</div>
            </div>
        </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Handle turns
        render_user_turn_reminder()
        render_user_input_section(controller)
        render_ai_turn(controller)
    
    # Show transcript
    render_debate_transcript()
    
    # Show final results
    render_final_results()
    
    # Reset button
    if st.session_state.debate_phase == 'complete':
        st.markdown("""
        <div style="text-align: center; margin: 3rem 0;">
            <div style="font-size: 1.25rem; color: #94a3b8; margin-bottom: 2rem;">
                Ready for another round of intellectual discourse? 🎯
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🔄 Start New Debate", type="primary", use_container_width=True, key="new_debate"):
                with st.spinner("🔄 Resetting debate chamber..."):
                    time.sleep(1)
                    for key in list(st.session_state.keys()):
                        del st.session_state[key]
                st.rerun()

if __name__ == "__main__":
    main()
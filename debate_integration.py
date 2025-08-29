# debate_integration.py

import os
import json
import asyncio
import threading
from typing import Dict, List, Optional, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DebateSystemIntegration:
    def __init__(self):
        self.is_running = False
        self.current_topic = None
        self.current_round = 1
        self.max_rounds = 3
        self.user_position = ""
        self.ai_position = ""
        self.user_score = 0
        self.ai_score = 0
        self.debate_history = []
        
    def start_debate(self, topic: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize a new debate session."""
        try:
            self.is_running = True
            self.current_topic = topic
            self.current_round = 1
            self.max_rounds = config.get("max_rounds", 3)
            self.user_score = 0
            self.ai_score = 0
            self.debate_history = []
            
            # Assign positions (simple logic - user gets "For", AI gets "Against")
            self.user_position = "For"
            self.ai_position = "Against"
            
            logger.info(f"Started debate on topic: {topic}")
            
            return {
                "success": True,
                "positions": {
                    "user_position": self.user_position,
                    "ai_position": self.ai_position
                }
            }
            
        except Exception as e:
            logger.error(f"Error starting debate: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def submit_user_argument(self, argument: str) -> Dict[str, Any]:
        """Process user argument and generate AI response with scoring."""
        try:
            if not self.is_running:
                return {
                    "success": False,
                    "error": "No active debate"
                }
            
            # Add user argument to history
            user_entry = {
                "speaker": "user",
                "round": self.current_round,
                "argument": argument,
                "score": None
            }
            self.debate_history.append(user_entry)
            
            # Generate AI response (simplified - in real implementation, use actual AI)
            ai_response = self._generate_ai_response(argument)
            
            # Score both arguments
            user_score = self._score_argument(argument, "user")
            ai_score = self._score_argument(ai_response, "ai")
            
            # Update scores
            self.user_score += user_score
            self.ai_score += ai_score
            
            # Add AI response to history
            ai_entry = {
                "speaker": "ai",
                "round": self.current_round,
                "argument": ai_response,
                "score": ai_score
            }
            self.debate_history.append(ai_entry)
            
            # Update user score in history
            user_entry["score"] = user_score
            
            # Check if round is complete
            round_complete = True
            
            # Move to next round if this one is complete
            if round_complete:
                self.current_round += 1
            
            # Generate feedback
            user_feedback = self._generate_feedback(argument, user_score, "user")
            ai_feedback = self._generate_feedback(ai_response, ai_score, "ai")
            
            logger.info(f"Processed round {self.current_round - 1}: User={user_score}, AI={ai_score}")
            
            return {
                "success": True,
                "user_score": user_score,
                "user_feedback": user_feedback,
                "ai_response": ai_response,
                "ai_score": ai_score,
                "ai_feedback": ai_feedback,
                "round_complete": round_complete
            }
            
        except Exception as e:
            logger.error(f"Error processing user argument: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _generate_ai_response(self, user_argument: str) -> str:
        """Generate AI response to user argument."""
        # Simplified response generation
        responses = [
            f"That's an interesting perspective. However, consider that {self._generate_counterpoint()}",
            f"While I understand your point, it's important to note that {self._generate_counterpoint()}",
            f"Your argument has merit, but let me offer an alternative viewpoint: {self._generate_counterpoint()}",
            f"I appreciate your position, though I'd like to highlight that {self._generate_counterpoint()}"
        ]
        
        import random
        return random.choice(responses)
    
    def _generate_counterpoint(self) -> str:
        """Generate a counterpoint for the AI response."""
        counterpoints = [
            "the evidence suggests a different conclusion when considering long-term effects.",
            "historical data shows that this approach has led to unintended consequences.",
            "there are multiple factors that need to be considered beyond the immediate impact.",
            "this perspective may not account for the broader societal implications.",
            "recent studies indicate that alternative approaches might be more effective."
        ]
        
        import random
        return random.choice(counterpoints)
    
    def _score_argument(self, argument: str, speaker: str) -> int:
        """Score an argument based on various criteria."""
        # Simplified scoring logic
        base_score = len(argument) // 10  # Longer arguments get higher base score
        quality_bonus = min(30, len(argument.split()) // 2)  # Word count bonus
        
        import random
        random_bonus = random.randint(-5, 5)  # Small random variation
        
        total_score = min(40, max(10, base_score + quality_bonus + random_bonus))
        
        return total_score
    
    def _generate_feedback(self, argument: str, score: int, speaker: str) -> str:
        """Generate feedback for an argument."""
        if score >= 35:
            return "Excellent argument! Strong logical structure, compelling evidence, and persuasive delivery."
        elif score >= 25:
            return "Good argument. Solid reasoning with room for improvement in evidence or persuasiveness."
        elif score >= 15:
            return "Adequate argument. Basic points made but needs stronger evidence and logical coherence."
        else:
            return "Needs improvement. Focus on developing clearer reasoning and supporting evidence."
    
    def end_debate(self) -> Dict[str, Any]:
        """End the current debate session."""
        self.is_running = False
        
        return {
            "success": True,
            "final_scores": {
                "user": self.user_score,
                "ai": self.ai_score
            },
            "total_rounds": self.current_round - 1
        }

# Global instance
debate_integration = DebateSystemIntegration()
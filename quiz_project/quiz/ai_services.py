"""
LangChain + Gemini AI Integration for Quiz Generation
"""
import os
import json
import re
from typing import List, Dict, Any
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from django.conf import settings


class APIKeyError(Exception):
    """Custom exception for missing API key"""
    pass


class QuizGenerator:
    """Quiz Generator using LangChain and Gemini AI"""
    
    def __init__(self):
        # Get API key from settings or environment variable
        self.api_key = getattr(settings, 'GOOGLE_API_KEY', None) or getattr(settings, 'GEMINI_API_KEY', None) or os.getenv('GOOGLE_API_KEY')
        
        # Check if API key is configured
        if not self.api_key or self.api_key == 'your_actual_gemini_api_key_here':
            raise APIKeyError("AI service not configured. Please contact administrator.")
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=self.api_key,
            temperature=0.7,
            convert_system_message_to_human=True
        )
    
    def generate_quiz_prompt(self, pdf_text: str, subject: str, difficulty: str, num_questions: int) -> str:
        """Generate the prompt for quiz creation"""
        
        prompt = f"""You are an expert quiz generator. Based on the following content from a {subject} PDF, generate {num_questions} multiple choice questions (MCQs) with {difficulty} difficulty level.

PDF CONTENT:
{pdf_text[:5000]}

INSTRUCTIONS:
1. Generate exactly {num_questions} questions
2. Each question must have 4 options (A, B, C, D)
3. Only ONE option should be correct
4. Provide a brief explanation for the correct answer
5. Questions should be clear, concise, and relevant to the subject
6. Format the output as JSON array

OUTPUT FORMAT:
{{
  "questions": [
    {{
      "question": "Question text here?",
      "options": {{
        "A": "Option A text",
        "B": "Option B text", 
        "C": "Option C text",
        "D": "Option D text"
      }},
      "correct_answer": "A",
      "explanation": "Explanation for why A is correct"
    }}
  ]
}}

Generate the quiz now:"""

        return prompt
    
    def parse_quiz_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse the AI response into structured quiz data"""
        
        # Try to extract JSON from response
        try:
            # Find JSON array in response
            json_match = re.search(r'\[[\s\S]*\]', response)
            if json_match:
                questions = json.loads(json_match.group())
            else:
                # Try parsing entire response as JSON
                data = json.loads(response)
                questions = data.get('questions', data)
            
            return questions
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Response: {response}")
            raise ValueError("Failed to parse quiz response from AI")
    
    def generate_questions(self, pdf_text: str, subject: str, difficulty: str, num_questions: int) -> List[Dict[str, Any]]:
        """Generate quiz questions using LangChain and Gemini"""
        
        prompt = self.generate_quiz_prompt(pdf_text, subject, difficulty, num_questions)
        
        # Call Gemini through LangChain
        response = self.llm.invoke(prompt)
        
        # Extract content from the response
        if hasattr(response, 'content'):
            response_text = response.content
        else:
            response_text = str(response)
        
        # Parse the response
        questions = self.parse_quiz_response(response_text)
        
        return questions


def generate_quiz_questions(pdf_text: str, subject: str, difficulty: str, num_questions: int) -> List[Dict[str, Any]]:
    """
    Main function to generate quiz questions
    This is called from Django views
    """
    generator = QuizGenerator()
    return generator.generate_questions(pdf_text, subject, difficulty, num_questions)


class PerformanceAnalyzer:
    """Performance Analyzer using LangChain and Gemini AI for advanced evaluation"""
    
    def __init__(self):
        # Get API key from settings or environment variable
        self.api_key = getattr(settings, 'GOOGLE_API_KEY', None) or getattr(settings, 'GEMINI_API_KEY', None) or os.getenv('GOOGLE_API_KEY')
        
        # Check if API key is configured
        if not self.api_key or self.api_key == 'your_actual_gemini_api_key_here':
            raise APIKeyError("AI service not configured. Please contact administrator.")
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=self.api_key,
            temperature=0.7,
            convert_system_message_to_human=True
        )
    
    def generate_feedback_prompt(self, subject: str, total_questions: int, 
                                 correct_count: int, wrong_count: int,
                                 user_answers: List[Dict], difficulty: str) -> str:
        """Generate the prompt for performance feedback"""
        
        percentage = (correct_count / total_questions * 100) if total_questions > 0 else 0
        
        # Prepare user answers summary
        answers_summary = ""
        for i, ans in enumerate(user_answers[:10], 1):  # Limit to first 10 for context
            status = "✓ Correct" if ans['is_correct'] else "✗ Wrong"
            answers_summary += f"{i}. {ans['question'][:100]}... - {status}\n"
        
        prompt = f"""You are an expert educational analyzer. Analyze the student's performance in a {subject} quiz and provide detailed feedback.

QUIZ DETAILS:
- Subject: {subject}
- Difficulty: {difficulty}
- Total Questions: {total_questions}
- Correct Answers: {correct_count}
- Wrong Answers: {wrong_count}
- Score Percentage: {percentage}%

USER ANSWERS:
{answers_summary}

INSTRUCTIONS:
Analyze the performance and provide feedback in the following JSON format:
{{
  "strength_analysis": "What the student did well - specific areas where they showed good understanding",
  "weakness_analysis": "What areas need improvement - topics or concepts they struggled with",
  "suggestions": "Specific suggestions for improvement - study tips, topics to review, etc."
}}

Be specific and helpful. The feedback should be actionable and educational.

Generate the feedback now:"""

        return prompt
    
    def parse_feedback_response(self, response: str) -> Dict[str, Any]:
        """Parse the AI response into structured feedback data"""
        
        try:
            # Try to find JSON in response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                feedback = json.loads(json_match.group())
                return {
                    'strength_analysis': feedback.get('strength_analysis', ''),
                    'weakness_analysis': feedback.get('weakness_analysis', ''),
                    'suggestions': feedback.get('suggestions', '')
                }
            else:
                # If no JSON found, create a basic response
                return {
                    'strength_analysis': 'Good effort on the quiz.',
                    'weakness_analysis': 'Some areas need improvement.',
                    'suggestions': 'Review the topics and try again.'
                }
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Response: {response}")
            return {
                'strength_analysis': 'Keep practicing to improve.',
                'weakness_analysis': 'Review the material more thoroughly.',
                'suggestions': 'Consider retaking the quiz after studying.'
            }
    
    def generate_feedback(self, subject: str, total_questions: int, 
                         correct_count: int, wrong_count: int,
                         user_answers: List[Dict], difficulty: str) -> Dict[str, Any]:
        """Generate performance feedback using LangChain and Gemini"""
        
        prompt = self.generate_feedback_prompt(
            subject, total_questions, correct_count, wrong_count,
            user_answers, difficulty
        )
        
        # Call Gemini through LangChain
        response = self.llm.invoke(prompt)
        
        # Extract content from the response
        if hasattr(response, 'content'):
            response_text = response.content
        else:
            response_text = str(response)
        
        # Parse the response
        feedback = self.parse_feedback_response(response_text)
        
        return feedback


def generate_performance_feedback(subject: str, total_questions: int,
                                  correct_count: int, wrong_count: int,
                                  user_answers: List[Dict], difficulty: str) -> Dict[str, Any]:
    """
    Main function to generate performance feedback
    This is called from Django views for advanced evaluation
    """
    analyzer = PerformanceAnalyzer()
    return analyzer.generate_feedback(
        subject, total_questions, correct_count, wrong_count,
        user_answers, difficulty
    )

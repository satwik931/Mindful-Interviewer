import os
import json
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def normalize_and_fuse_sentiment(face_analysis, voice_analysis, text_analysis):
    """
    Combines analysis from all modalities into a single sentiment score.
    Returns a score between -1.0 (negative) and +1.0 (positive).
    """
    emotion_scores = {
        'happy': 0.8, 'surprise': 0.5, 'neutral': 0.1,
        'sad': -0.6, 'angry': -0.7, 'fear': -0.8, 'disgust': -0.9
    }
    dominant_emotion = face_analysis.get('dominant_emotion', 'neutral')
    face_score = emotion_scores.get(dominant_emotion, 0.0)

    pitch = voice_analysis.get('average_pitch_hz', 150)
    pitch_deviation = abs(pitch - 150) / 150
    voice_score = -pitch_deviation

    filler_ratio = text_analysis.get('filler_ratio', 0.0)
    text_score = 1 - (filler_ratio / 0.15)
    text_score = np.clip(text_score, -1.0, 1.0)

    weights = {'face': 0.5, 'voice': 0.2, 'text': 0.3}

    fused_score = (face_score * weights['face'] +
                   voice_score * weights['voice'] +
                   text_score * weights['text'])
    
    fused_score = np.clip(fused_score, -1.0, 1.0)

    print(f"--- Fused Sentiment Analysis ---")
    print(f"Face Score: {face_score:.2f} (Emotion: {dominant_emotion})")
    print(f"Voice Score: {voice_score:.2f} (Pitch: {pitch:.0f}Hz)")
    print(f"Text Score: {text_score:.2f} (Filler Ratio: {filler_ratio:.2f})")
    print(f"Final Fused Score: {fused_score:.2f}")
    
    return fused_score

def generate_adaptive_question(conversation_history, fused_sentiment_score):
    """
    Generates the next interview question using the Google Gemini API.
    """
    sentiment_description = "neutral"
    if fused_sentiment_score > 0.4:
        sentiment_description = "confident and positive"
    elif fused_sentiment_score < -0.3:
        sentiment_description = "nervous and hesitant"

    prompt = f"""
    You are an expert, friendly, and encouraging HR interviewer named 'Gemini'.
    Your goal is to assess a candidate for a 'Software Engineer' role.
    The candidate's current emotional state is perceived as: {sentiment_description}.

    RULES:
    - If the candidate seems nervous, ask a simpler, rapport-building question.
    - If the candidate seems confident, ask a more challenging follow-up or a behavioral question.
    - Keep your questions concise and professional.
    - NEVER break character.
    - ALWAYS respond in a valid JSON format with two keys: "question_text" and "suggested_avatar_emotion".
    - For "suggested_avatar_emotion", choose from: 'neutral', 'smiling', 'encouraging_nod', 'thinking'.

    Here is the conversation so far:
    {conversation_history}

    Based on all the rules, the conversation, and the candidate's emotional state, generate the next single interview question.
    """
    
    try:
        # The only change is on the next line
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        
        response = model.generate_content(prompt)
        
        response_text = response.text.strip().replace("```json", "").replace("```", "")
        
        response_json = json.loads(response_text)
        print(f"LLM Response: {response_json}")
        return response_json

    except Exception as e:
        print(f"Error communicating with Google Gemini API: {e}")
        return None
import cv2
import speech_recognition as sr
import numpy as np
import time # <--- ADD THIS LINE
from deepface import DeepFace
import os
import uuid # To create unique filenames for audio

# --- Import our custom modules ---
from core_logic import normalize_and_fuse_sentiment, generate_adaptive_question
from output_engine import speak_with_animation
# We need the analysis functions too
from analyze_audio import analyze_voice_tone
from analyze_text import analyze_filler_words

# --- Main Application Logic ---
def main():
    # 1. INITIALIZATION
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        return

    r = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        # Adjust for ambient noise once at the beginning
        print("Adjusting for ambient noise... Please wait.")
        r.adjust_for_ambient_noise(source, duration=2)
        print("Ready to begin.")

    # Initialize conversation history
    conversation_history = []
    
    # Create a window for the avatar
    cv2.namedWindow('AI Interviewer', cv2.WINDOW_NORMAL)
    
    # 2. START THE INTERVIEW
    # First turn: The AI greets the user and asks the first question.
    initial_greeting = "Hello and welcome to the interview. Let's start with an easy one. Can you tell me a little bit about yourself?"
    llm_response = {
        'question_text': initial_greeting,
        'suggested_avatar_emotion': 'smiling'
    }
    conversation_history.append({'role': 'interviewer', 'content': initial_greeting})
    
    # Main conversation loop
    while True:
        # 3. INTERVIEWER'S TURN (SPEAKING)
        speak_with_animation(
            text=llm_response['question_text'],
            emotion=llm_response.get('suggested_avatar_emotion', 'neutral')
        )
        
        # 4. CANDIDATE'S TURN (LISTENING AND ANALYSIS)
        # Display neutral avatar while listening
        neutral_img = cv2.imread("avatar_images/neutral.png")
        if neutral_img is not None:
            cv2.imshow('AI Interviewer', neutral_img)
        cv2.waitKey(1)

        # Listen for the user's response
        print("Listening for your response...")
        try:
            with mic as source:
                audio = r.listen(source, timeout=10, phrase_time_limit=30)
        except sr.WaitTimeoutError:
            print("Listening timed out. No speech detected.")
            # Ask them to repeat
            llm_response = {'question_text': "I'm sorry, I didn't hear anything. Could you please answer the question?", 'suggested_avatar_emotion': 'thinking'}
            continue # Skip the rest of the loop and go to the AI's speaking turn

        # Display thinking avatar while processing
        thinking_img = cv2.imread("avatar_images/thinking.png")
        if thinking_img is not None:
            cv2.imshow('AI Interviewer', thinking_img)
        cv2.waitKey(1)
        
        print("Processing your response...")
        
        # Capture a single frame for facial analysis with a retry mechanism
        frame = None
        for i in range(3): # Try up to 3 times to get a frame
            ret, frame = cap.read()
            if ret:
                break # Success
            time.sleep(0.1) # Wait a bit before retrying

        if frame is None:
            print("Error: Failed to capture frame from webcam after multiple attempts.")
            print("Please ensure your webcam is not in use by another application and that terminal has camera permissions in your System Settings.")
            face_analysis = {'dominant_emotion': 'neutral'} # Use a dummy analysis
        else:
            try:
                # Perform facial analysis
                face_analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)[0]
            except Exception as e:
                print(f"Facial analysis error: {e}")
                face_analysis = {'dominant_emotion': 'neutral'} # Default on error

        # Save audio to a temporary file for analysis
        temp_audio_file = f"temp_{uuid.uuid4()}.wav"
        with open(temp_audio_file, "wb") as f:
            f.write(audio.get_wav_data())

        # Perform multimodal analysis
        try:
            # Speech-to-Text
            candidate_text = r.recognize_google(audio)
            print(f"You said: {candidate_text}")
            
            # Voice and Text Analysis
            voice_analysis = analyze_voice_tone(temp_audio_file)
            text_analysis = analyze_filler_words(candidate_text)
            
            # Clean up the temp audio file
            os.remove(temp_audio_file)

        except sr.UnknownValueError:
            print("Could not understand the audio.")
            os.remove(temp_audio_file)
            llm_response = {'question_text': "I'm sorry, I couldn't quite understand that. Could you please repeat yourself?", 'suggested_avatar_emotion': 'thinking'}
            continue
        except Exception as e:
            print(f"An error occurred during analysis: {e}")
            if os.path.exists(temp_audio_file):
                os.remove(temp_audio_file)
            # Generic error message
            llm_response = {'question_text': "I encountered a small issue. Let's try that again.", 'suggested_avatar_emotion': 'thinking'}
            continue
        
        # Add candidate's response to history
        conversation_history.append({'role': 'candidate', 'content': candidate_text})

        # Check for exit condition
        if "goodbye" in candidate_text.lower() or "end interview" in candidate_text.lower():
            break
            
        # 5. AI'S TURN (THINKING)
        # Fuse sentiments
        fused_sentiment = normalize_and_fuse_sentiment(face_analysis, voice_analysis, text_analysis)
        
        # Generate the next question
        llm_response = generate_adaptive_question(conversation_history, fused_sentiment)
        
        if llm_response is None:
            print("Failed to get response from LLM. Ending interview.")
            break
            
        # Add AI's new question to history
        conversation_history.append({'role': 'interviewer', 'content': llm_response['question_text']})
        
        # The loop repeats

    # 6. END OF INTERVIEW
    final_goodbye = "Thank you for your time. That concludes our interview. We will be in touch with you shortly. Goodbye."
    speak_with_animation(final_goodbye, 'smiling')
    
    cap.release()
    cv2.destroyAllWindows()
    print("Interview finished.")

if __name__ == "__main__":
    main()
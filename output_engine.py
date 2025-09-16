import cv2
import os
from gtts import gTTS
from mutagen.mp3 import MP3
import pygame
import time

def speak_with_animation(text, emotion, audio_file="response.mp3"):
    """
    1. Converts text to speech.
    2. Plays the speech using pygame while showing a speaking animation.
    3. Reverts to a neutral face after speaking.
    """
    try:
        # --- 1. Generate Audio ---
        tts = gTTS(text=text, lang='en')
        tts.save(audio_file)
        
        # --- 2. Initialize Pygame Mixer and Load Audio ---
        pygame.mixer.init()
        pygame.mixer.music.load(audio_file)

        print(f"Avatar says: {text}")

        # --- 3. Play Audio with Speaking Animation ---
        speaking_img = cv2.imread(f"avatar_images/speaking.png")
        if speaking_img is None: speaking_img = cv2.imread("avatar_images/neutral.png")
        
        cv2.imshow('AI Interviewer', speaking_img)
        cv2.waitKey(1) # Refresh window

        pygame.mixer.music.play()

        # Wait for the audio to finish playing
        while pygame.mixer.music.get_busy():
            # We add a brief wait to prevent this loop from consuming 100% CPU
            pygame.time.Clock().tick(10)
            # We still need to process OpenCV events to keep the window responsive
            cv2.waitKey(1)

        # --- 4. Clean up and Revert to Neutral ---
        pygame.mixer.quit()
        
        # Short pause after speaking
        time.sleep(0.3) 
        
        neutral_img = cv2.imread("avatar_images/neutral.png")
        cv2.imshow('AI Interviewer', neutral_img)
        cv2.waitKey(1)

        os.remove(audio_file)

    except Exception as e:
        print(f"Error in speak_with_animation: {e}")
        # Ensure pygame mixer is uninitialized on error
        if pygame.mixer.get_init():
            pygame.mixer.quit()

# --- Example Usage (for testing this file directly) ---
if __name__ == '__main__':
    cv2.namedWindow('AI Interviewer', cv2.WINDOW_NORMAL)
    speak_with_animation(
        "This is a test using Pygame for audio playback. It should be much more reliable.",
        "smiling"
    )
    cv2.destroyAllWindows()
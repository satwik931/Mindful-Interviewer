import cv2
import time

def display_avatar_emotion(emotion, duration_sec=2):
    """
    Displays an avatar image based on the given emotion for a set duration.
    """
    image_path = f"avatar_images/{emotion}.png"

    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not load image at {image_path}")
            # Fallback to neutral if image not found
            img = cv2.imread("avatar_images/neutral.png")

        cv2.imshow('AI Interviewer', img)

        # Wait for the specified duration (or until a key is pressed)
        # The waitKey value is in milliseconds
        cv2.waitKey(int(duration_sec * 1000))

    except Exception as e:
        print(f"Error displaying avatar: {e}")

# --- Example Usage ---
if __name__ == '__main__':
    print("Displaying neutral face for 2 seconds...")
    display_avatar_emotion('neutral', duration_sec=2)

    print("Displaying smiling face for 2 seconds...")
    display_avatar_emotion('smiling', duration_sec=2)

    print("Displaying thinking face for 2 seconds...")
    display_avatar_emotion('thinking', duration_sec=2)

    cv2.destroyAllWindows()
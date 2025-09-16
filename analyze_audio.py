import librosa
import numpy as np
import soundfile as sf

def analyze_voice_tone(audio_file_path):
    """
    Analyzes the pitch and energy of an audio file.
    Returns a dictionary with the analysis.
    """
    try:
        # Load the audio file
        y, sr = librosa.load(audio_file_path)

        # Calculate Root Mean Square (RMS) for energy
        rms_energy = np.mean(librosa.feature.rms(y=y))

        # Estimate pitch using a pitch tracking algorithm
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        # Select pitches with high confidence (non-zero magnitude)
        confident_pitches = pitches[magnitudes > 0]
        average_pitch = np.mean(confident_pitches) if len(confident_pitches) > 0 else 0

        # For now, we'll just return the raw values.
        # Later, we can classify these into "calm", "energetic", etc.
        analysis = {
            'average_pitch_hz': float(average_pitch),
            'energy': float(rms_energy)
        }

        print(f"Voice Analysis: {analysis}")
        return analysis

    except Exception as e:
        print(f"Error analyzing audio: {e}")
        return None

# Example Usage:
if __name__ == '__main__':
    # To test this, you need an audio file. You can record one using any tool.
    # For example, let's create a dummy file for testing purposes.
    # In a real scenario, this would be the file recorded from the user.
    sr_test = 22050
    duration = 3
    frequency = 440
    t = np.linspace(0., duration, int(sr_test * duration))
    amplitude = np.iinfo(np.int16).max * 0.5
    data = amplitude * np.sin(2. * np.pi * frequency * t)

    dummy_file = 'test_audio.wav'
    sf.write(dummy_file, data.astype(np.int16), sr_test)

    print(f"Analyzing dummy file: {dummy_file}")
    analyze_voice_tone(dummy_file)
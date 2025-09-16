def analyze_filler_words(text):
    """
    Counts the number of filler words in a transcribed text.
    """
    FILLER_WORDS = [
        'uh', 'um', 'ah', 'er', 'like', 'so', 'you know', 
        'basically', 'actually', 'i mean', 'right'
    ]
    
    words = text.lower().split()
    filler_count = 0
    for word in words:
        if word in FILLER_WORDS:
            filler_count += 1
            
    analysis = {
        'word_count': len(words),
        'filler_count': filler_count,
        'filler_ratio': filler_count / len(words) if len(words) > 0 else 0
    }
    
    print(f"Text Analysis: {analysis}")
    return analysis

# Example Usage:
if __name__ == '__main__':
    sample_text_confident = "I have three years of experience with Python and machine learning."
    sample_text_nervous = "Um, so, like, I have, you know, basically three years of, uh, experience."

    print("Analyzing confident statement:")
    analyze_filler_words(sample_text_confident)
    
    print("\nAnalyzing nervous statement:")
    analyze_filler_words(sample_text_nervous)
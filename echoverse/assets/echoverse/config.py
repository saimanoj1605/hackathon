# Configuration for local Granite model
MODEL_CONFIG = {
    # Available Granite models (choose based on your hardware)
    "models": {
        "granite-3b": "ibm-granite/granite-3.0-3b-a800m-instruct",
        "granite-2b": "ibm-granite/granite-3.1-2b-instruct", 
        "granite-8b": "ibm-granite/granite-3.0-8b-instruct"  # Requires more RAM
    },
    
    # Default model (3B is good balance of performance/resources)
    "default_model": "granite-3b",
    
    # Generation parameters
    "generation_params": {
        "max_new_tokens": 300,
        "temperature": 0.7,
        "do_sample": True,
        "pad_token_id": 50256
    }
}

# TTS Configuration
TTS_CONFIG = {
    "engine": "pyttsx3",  # Options: "pyttsx3", "gtts"
    "voice_speed": 150,   # Words per minute
    "voice_volume": 0.8   # 0.0 to 1.0
}

# Tone prompts for text rewriting
TONE_PROMPTS = {
    "Neutral": "Rewrite the following text in a clear, neutral tone while preserving all key information and meaning:",
    "Suspenseful": "Rewrite the following text in a suspenseful, dramatic tone that builds tension and excitement:",
    "Inspiring": "Rewrite the following text in an inspiring, motivational tone that energizes and uplifts the reader:"
}
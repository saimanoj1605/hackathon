import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pyttsx3
from gtts import gTTS
import io
import tempfile
import os
from config import MODEL_CONFIG, TTS_CONFIG, TONE_PROMPTS
import threading
import time

# Page configuration
st.set_page_config(
    page_title="EchoVerse - Local AI Audiobook Creator",
    page_icon="🎧",
    layout="wide"
)

# Initialize session state
if 'narrations' not in st.session_state:
    st.session_state.narrations = []
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = None

@st.cache_resource
def load_granite_model(model_name="granite-3b"):
    """Load IBM Granite model locally"""
    model_id = MODEL_CONFIG["models"][model_name]
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def rewrite_text_with_granite(text, tone):
    """Rewrite text using local Granite model"""
    if st.session_state.model is None or st.session_state.tokenizer is None:
        return text
    
    try:
        # Create prompt
        prompt = f"{TONE_PROMPTS[tone]}\n\nOriginal text: {text}\n\nRewritten text:"
        
        # Tokenize
        inputs = st.session_state.tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=1024
        )
        
        # Generate
        with torch.no_grad():
            outputs = st.session_state.model.generate(
                inputs.input_ids,
                **MODEL_CONFIG["generation_params"],
                attention_mask=inputs.attention_mask
            )
        
        # Decode
        generated_text = st.session_state.tokenizer.decode(
            outputs[0], 
            skip_special_tokens=True
        )
        
        # Extract only the rewritten part
        if "Rewritten text:" in generated_text:
            rewritten = generated_text.split("Rewritten text:")[-1].strip()
        else:
            rewritten = generated_text[len(prompt):].strip()
        
        return rewritten if rewritten else text
        
    except Exception as e:
        st.error(f"Error rewriting text: {str(e)}")
        return text

def generate_audio_pyttsx3(text, voice_name="default"):
    """Generate audio using pyttsx3 (offline)"""
    try:
        engine = pyttsx3.init()
        
        # Configure voice
        voices = engine.getProperty('voices')
        if voices:
            if voice_name == "female" and len(voices) > 1:
                engine.setProperty('voice', voices[1].id)
            else:
                engine.setProperty('voice', voices[0].id)
        
        # Configure speech rate and volume
        engine.setProperty('rate', TTS_CONFIG["voice_speed"])
        engine.setProperty('volume', TTS_CONFIG["voice_volume"])
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            engine.save_to_file(text, tmp_file.name)
            engine.runAndWait()
            
            # Read the audio file
            with open(tmp_file.name, 'rb') as audio_file:
                audio_data = audio_file.read()
            
            # Clean up
            os.unlink(tmp_file.name)
            
        return audio_data
        
    except Exception as e:
        st.error(f"Error generating audio with pyttsx3: {str(e)}")
        return None

def generate_audio_gtts(text, language='en'):
    """Generate audio using Google Text-to-Speech (requires internet)"""
    try:
        tts = gTTS(text=text, lang=language, slow=False)
        
        # Save to bytes buffer
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        
        return audio_buffer.read()
        
    except Exception as e:
        st.error(f"Error generating audio with gTTS: {str(e)}")
        return None

def generate_audio(text, voice="default"):
    """Generate audio using configured TTS engine"""
    if TTS_CONFIG["engine"] == "gtts":
        return generate_audio_gtts(text)
    else:
        return generate_audio_pyttsx3(text, voice)

def add_custom_css():
    st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    
    .stTitle {
        color: #2E86AB;
        text-align: center;
        font-size: 3rem !important;
        margin-bottom: 1rem;
    }
    
    .stSubheader {
        color: #A23B72;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .model-status {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    
    .model-loaded {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    
    .model-loading {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #2E86AB, #A23B72);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    add_custom_css()
    
    # Title and header
    st.title("🎧 EchoVerse Local")
    st.subheader("Transform Text into Expressive Audiobooks with Local AI")
    st.markdown("*Powered by IBM Granite 3B - No internet required for AI processing!*")
    
    # Model loading section
    st.header("🤖 AI Model Setup")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_model = st.selectbox(
            "Choose Granite Model:",
            options=list(MODEL_CONFIG["models"].keys()),
            index=0,
            help="3B model is recommended for most computers. 8B requires more RAM."
        )
    
    with col2:
        if st.button("Load Model", type="primary"):
            with st.spinner(f"Loading {selected_model} model... This may take a few minutes on first run."):
                model, tokenizer = load_granite_model(selected_model)
                if model and tokenizer:
                    st.session_state.model = model
                    st.session_state.tokenizer = tokenizer
                    st.session_state.model_loaded = True
                    st.success("Model loaded successfully!")
                else:
                    st.error("Failed to load model")
    
    # Model status
    if st.session_state.model_loaded:
        st.markdown('<div class="model-status model-loaded">✅ AI Model Ready</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="model-status model-loading">⏳ Please load AI model first</div>', unsafe_allow_html=True)
        st.warning("You need to load the AI model before generating audiobooks.")
        return
    
    # Input section
    st.header("📝 Input Your Content")
    
    # File upload option
    uploaded_file = st.file_uploader(
        "Upload a text file", 
        type=['txt'],
        help="Upload a .txt file to convert to audiobook"
    )
    
    # Text input option
    text_input = st.text_area(
        "Or paste your text here:",
        height=200,
        placeholder="Enter the text you want to convert to an audiobook..."
    )
    
    # Determine input text
    input_text = ""
    if uploaded_file is not None:
        input_text = str(uploaded_file.read(), "utf-8")
        st.success("File uploaded successfully!")
    elif text_input:
        input_text = text_input
    
    if input_text:
        # Truncate if too long
        if len(input_text) > 2000:
            input_text = input_text[:2000]
            st.warning("Text truncated to 2000 characters for optimal processing.")
        
        st.info(f"Text length: {len(input_text)} characters")
        
        # Configuration options
        st.header("⚙️ Audio Configuration")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            tone = st.selectbox(
                "Select Tone:",
                ["Neutral", "Suspenseful", "Inspiring"],
                help="Choose how you want the text to be rewritten"
            )
        
        with col2:
            tts_engine = st.selectbox(
                "TTS Engine:",
                ["pyttsx3 (Offline)", "gTTS (Online)"],
                help="pyttsx3 works offline, gTTS needs internet but sounds better"
            )
        
        with col3:
            voice_type = st.selectbox(
                "Voice Type:",
                ["default", "female"],
                help="Available voices depend on your system"
            )
        
        # Update TTS config
        TTS_CONFIG["engine"] = "gtts" if "gTTS" in tts_engine else "pyttsx3"
        
        # Generate button
        if st.button("🎵 Generate Audiobook", type="primary"):
            
            # Step 1: Rewrite text
            with st.spinner("Rewriting text with AI..."):
                rewritten_text = rewrite_text_with_granite(input_text, tone)
            
            # Step 2: Generate audio
            with st.spinner("Generating audio..."):
                audio_data = generate_audio(rewritten_text, voice_type)
            
            if audio_data:
                # Display results
                st.header("📊 Results")
                
                # Side-by-side text comparison
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Original Text")
                    st.text_area("", value=input_text, height=300, disabled=True, key="orig")
                
                with col2:
                    st.subheader(f"Rewritten Text ({tone})")
                    st.text_area("", value=rewritten_text, height=300, disabled=True, key="rewritten")
                
                # Audio playback and download
                st.header("🎧 Your Audiobook")
                
                # Determine audio format
                audio_format = "audio/wav" if TTS_CONFIG["engine"] == "pyttsx3" else "audio/mp3"
                file_extension = "wav" if TTS_CONFIG["engine"] == "pyttsx3" else "mp3"
                
                st.audio(audio_data, format=audio_format)
                
                # Download button
                st.download_button(
                    label=f"📥 Download {file_extension.upper()}",
                    data=audio_data,
                    file_name=f"audiobook_{tone.lower()}_{voice_type}.{file_extension}",
                    mime=audio_format
                )
                
                # Save to session state
                st.session_state.narrations.append({
                    'text': rewritten_text,
                    'tone': tone,
                    'voice': voice_type,
                    'audio': audio_data,
                    'format': audio_format,
                    'extension': file_extension,
                    'timestamp': str(len(st.session_state.narrations) + 1)
                })
                
                st.success("Audiobook generated successfully!")
        
        # Past narrations section
        if st.session_state.narrations:
            st.header("📚 Past Narrations")
            
            with st.expander(f"View All ({len(st.session_state.narrations)} narrations)"):
                for i, narration in enumerate(reversed(st.session_state.narrations)):
                    st.subheader(f"Narration {len(st.session_state.narrations) - i}")
                    st.write(f"**Tone:** {narration['tone']} | **Voice:** {narration['voice']}")
                    
                    # Show text in expandable section
                    with st.expander("View Text"):
                        st.write(narration['text'])
                    
                    # Audio player
                    st.audio(narration['audio'], format=narration['format'])
                    
                    # Download button
                    st.download_button(
                        label=f"📥 Re-download {narration['extension'].upper()}",
                        data=narration['audio'],
                        file_name=f"audiobook_{narration['tone'].lower()}_{narration['voice']}_{narration['timestamp']}.{narration['extension']}",
                        mime=narration['format'],
                        key=f"download_{i}"
                    )
                    st.divider()

    # System information
    with st.sidebar:
        st.header("📊 System Info")
        st.write(f"**GPU Available:** {'✅ Yes' if torch.cuda.is_available() else '❌ No (CPU only)'}")
        st.write(f"**Model Loaded:** {'✅ Yes' if st.session_state.model_loaded else '❌ No'}")
        st.write(f"**TTS Engine:** {TTS_CONFIG['engine']}")
        
        if st.session_state.model_loaded:
            device = next(st.session_state.model.parameters()).device
            st.write(f"**Model Device:** {device}")
        
        st.header("💡 Tips")
        st.write("• First model load takes time")
        st.write("• 3B model: ~6GB RAM needed")  
        st.write("• 8B model: ~16GB RAM needed")
        st.write("• GPU greatly speeds up processing")
        st.write("• gTTS requires internet connection")

if __name__ == "__main__":
    main()
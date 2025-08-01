import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from gtts import gTTS
import io
import tempfile
import os
import json

# Configuration (since we don't have the config.py file)
MODEL_CONFIG = {
    "models": {
        "granite-3b": "ibm-granite/granite-3b-code-base",
        "granite-8b": "ibm-granite/granite-8b-code-base"
    },
    "generation_params": {
        "max_new_tokens": 512,
        "temperature": 0.7,
        "do_sample": True,
        "pad_token_id": None
    }
}

TTS_CONFIG = {
    "engine": "gtts",
    "voice_speed": 150,
    "voice_volume": 0.9
}

TONE_PROMPTS = {
    "Neutral": "Rewrite the following text in a clear, neutral tone suitable for audiobook narration:",
    "Suspenseful": "Rewrite the following text with suspenseful, engaging language that builds tension:",
    "Inspiring": "Rewrite the following text in an inspiring, motivational tone that uplifts the reader:"
}

# Global variables to store model
model = None
tokenizer = None
model_loaded = False

def load_granite_model(model_name="granite-3b"):
    """Load IBM Granite model locally"""
    global model, tokenizer, model_loaded
    
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
        
        model_loaded = True
        return "‚úÖ Model loaded successfully!"
    except Exception as e:
        model_loaded = False
        return f"‚ùå Error loading model: {str(e)}"

def rewrite_text_with_granite(text, tone):
    """Rewrite text using local Granite model"""
    global model, tokenizer, model_loaded
    
    if not model_loaded or model is None or tokenizer is None:
        return text
    
    try:
        # Create prompt
        prompt = f"{TONE_PROMPTS[tone]}\n\nOriginal text: {text}\n\nRewritten text:"
        
        # Tokenize
        inputs = tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=1024
        )
        
        # Set pad_token_id for generation
        generation_params = MODEL_CONFIG["generation_params"].copy()
        generation_params["pad_token_id"] = tokenizer.pad_token_id
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                **generation_params,
                attention_mask=inputs.attention_mask
            )
        
        # Decode
        generated_text = tokenizer.decode(
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
        return f"Error rewriting text: {str(e)}"

def generate_audio_gtts(text, language='en'):
    """Generate audio using Google Text-to-Speech"""
    try:
        tts = gTTS(text=text, lang=language, slow=False)
        
        # Save to temporary file and return path
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
            tts.save(tmp_file.name)
            return tmp_file.name
        
    except Exception as e:
        return None

def process_audiobook(input_text, uploaded_file, tone, model_choice):
    """Main processing function"""
    global model_loaded
    
    # Check if model is loaded
    if not model_loaded:
        return (
            "‚ùå Please load the AI model first!",
            None,
            None,
            "Please click 'Load Model' button first."
        )
    
    # Determine input text
    text_to_process = ""
    if uploaded_file is not None:
        try:
            # Read uploaded file
            content = uploaded_file.read()
            if isinstance(content, bytes):
                text_to_process = content.decode('utf-8')
            else:
                text_to_process = str(content)
        except Exception as e:
            return f"Error reading file: {str(e)}", None, None, ""
    elif input_text:
        text_to_process = input_text
    else:
        return "Please provide text input or upload a file.", None, None, ""
    
    # Truncate if too long
    if len(text_to_process) > 2000:
        text_to_process = text_to_process[:2000]
        status_msg = "‚ö†Ô∏è Text truncated to 2000 characters for optimal processing."
    else:
        status_msg = f"‚úÖ Processing {len(text_to_process)} characters."
    
    # Rewrite text with AI
    try:
        rewritten_text = rewrite_text_with_granite(text_to_process, tone)
    except Exception as e:
        return f"Error in text rewriting: {str(e)}", None, None, ""
    
    # Generate audio
    try:
        audio_file_path = generate_audio_gtts(rewritten_text)
        if audio_file_path is None:
            return status_msg, text_to_process, rewritten_text, "‚ùå Failed to generate audio."
    except Exception as e:
        return status_msg, text_to_process, rewritten_text, f"Error generating audio: {str(e)}"
    
    return (
        status_msg,
        text_to_process,
        rewritten_text, 
        audio_file_path
    )

def get_model_status():
    """Get current model status"""
    global model_loaded
    if model_loaded:
        device = "GPU" if torch.cuda.is_available() else "CPU"
        return f"‚úÖ Model loaded on {device}"
    else:
        return "‚ùå Model not loaded"

# Create Gradio interface
def create_interface():
    with gr.Blocks(
        title="EchoVerse - Local AI Audiobook Creator",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            font-family: 'Arial', sans-serif;
        }
        .main-header {
            text-align: center;
            color: #2E86AB;
            margin-bottom: 20px;
        }
        .status-box {
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }
        """
    ) as demo:
        
        # Header
        gr.HTML("""
        <div class="main-header">
            <h1>Ìæß EchoVerse Local</h1>
            <h3>Transform Text into Expressive Audiobooks with Local AI</h3>
            <p><i>Powered by IBM Granite 3B - No internet required for AI processing!</i></p>
        </div>
        """)
        
        # Model Setup Section
        with gr.Group():
            gr.HTML("<h2>Ì¥ñ AI Model Setup</h2>")
            
            with gr.Row():
                model_choice = gr.Dropdown(
                    choices=list(MODEL_CONFIG["models"].keys()),
                    value="granite-3b",
                    label="Choose Granite Model",
                    info="3B model is recommended for most computers. 8B requires more RAM."
                )
                
                load_btn = gr.Button("Load Model", variant="primary")
            
            model_status = gr.Textbox(
                label="Model Status",
                value="‚ùå Model not loaded",
                interactive=False
            )
        
        # Input Section
        with gr.Group():
            gr.HTML("<h2>Ì≥ù Input Your Content</h2>")
            
            uploaded_file = gr.File(
                label="Upload a text file",
                file_types=[".txt"],
                type="binary"
            )
            
            input_text = gr.Textbox(
                label="Or paste your text here:",
                lines=8,
                placeholder="Enter the text you want to convert to an audiobook...",
                max_lines=15
            )
        
        # Configuration Section
        with gr.Group():
            gr.HTML("<h2>‚öôÔ∏è Audio Configuration</h2>")
            
            with gr.Row():
                tone = gr.Dropdown(
                    choices=["Neutral", "Suspenseful", "Inspiring"],
                    value="Neutral",
                    label="Select Tone",
                    info="Choose how you want the text to be rewritten"
                )
        
        # Generate Button
        generate_btn = gr.Button("Ìæµ Generate Audiobook", variant="primary", size="lg")
        
        # Results Section
        with gr.Group():
            gr.HTML("<h2>Ì≥ä Results</h2>")
            
            status_output = gr.Textbox(
                label="Status",
                interactive=False
            )
            
            with gr.Row():
                original_text = gr.Textbox(
                    label="Original Text",
                    lines=10,
                    interactive=False
                )
                
                rewritten_text = gr.Textbox(
                    label="Rewritten Text",
                    lines=10,
                    interactive=False
                )
            
            # Audio Output
            gr.HTML("<h2>Ìæß Your Audiobook</h2>")
            audio_output = gr.Audio(
                label="Generated Audiobook",
                type="filepath"
            )
        
        # System Info
        with gr.Group():
            gr.HTML("<h2>Ì≥ä System Info</h2>")
            
            system_info = gr.HTML(f"""
            <div>
                <p><strong>GPU Available:</strong> {'‚úÖ Yes' if torch.cuda.is_available() else '‚ùå No (CPU only)'}</p>
                <p><strong>TTS Engine:</strong> {TTS_CONFIG['engine']}</p>
            </div>
            
            <h3>Ì≤° Tips</h3>
            <ul>
                <li>First model load takes time</li>
                <li>3B model: ~6GB RAM needed</li>
                <li>8B model: ~16GB RAM needed</li>
                <li>GPU greatly speeds up processing</li>
                <li>gTTS requires internet connection</li>
            </ul>
            """)
        
        # Event handlers
        load_btn.click(
            fn=load_granite_model,
            inputs=[model_choice],
            outputs=[model_status]
        )
        
        generate_btn.click(
            fn=process_audiobook,
            inputs=[input_text, uploaded_file, tone, model_choice],
            outputs=[status_output, original_text, rewritten_text, audio_output]
        )
    
    return demo

# Launch the app
if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )

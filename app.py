from flask import Flask, request, render_template, jsonify
import os
from werkzeug.utils import secure_filename
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
import soundfile as sf
import gc
import requests
import time
from groq import Groq
import json

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp3', 'wav'}
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# NLLB Configuration
NLLB_API_URL = "https://api-inference.huggingface.co/models/facebook/nllb-200-distilled-600M"
NLLB_HEADERS = {"Authorization": "Bearer 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"}

# Groq Configuration
GROQ_API_KEY = "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"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Audio Processing Functions
def split_audio(audio_file, chunk_length_sec=30):
    try:
        audio, sr = librosa.load(audio_file, sr=16000, mono=True)
        total_duration = len(audio) / sr
        chunks = []
        num_chunks = int(total_duration // chunk_length_sec) + 1
        for i in range(num_chunks):
            start_sample = i * chunk_length_sec * sr
            end_sample = min((i + 1) * chunk_length_sec * sr, len(audio))
            chunk = audio[start_sample:end_sample]
            chunk_name = os.path.join(app.config['UPLOAD_FOLDER'], f"chunk_{i}.wav")
            sf.write(chunk_name, chunk, sr)
            chunks.append(chunk_name)
        return chunks
    except Exception as e:
        print(f"Error splitting audio: {str(e)}")
        return []

def transcribe_audio(audio_file, processor, model, device):
    try:
        audio, sr = librosa.load(audio_file, sr=16000, mono=True)
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt").to(device)
        
        with torch.no_grad():
            generated_ids = model.generate(
                inputs.input_features,
                max_length=448,
                language="kn",
                task="transcribe",
                forced_decoder_ids=None
            )
        
        transcribed_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return transcribed_text
    except Exception as e:
        print(f"Error in transcription: {str(e)}")
        return None

# NLLB Translation Functions
def query_nllb(text, src_lang, tgt_lang, retries=3, delay=10):
    input_data = {
        "inputs": text,
        "parameters": {
            "src_lang": src_lang,
            "tgt_lang": tgt_lang
        }
    }
    
    for attempt in range(retries):
        try:
            response = requests.post(NLLB_API_URL, headers=NLLB_HEADERS, json=input_data)
            if response.status_code == 200:
                return response.json()[0]["translation_text"]
            elif response.status_code == 503:
                time.sleep(delay)
            else:
                print(f"Error: {response.status_code}")
                break
        except Exception as e:
            print(f"Request error: {e}")
    return None

# Groq QA Function
class StoryQASystem:
    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)
        
    def find_answer(self, question, directory="translations"):
        story_files = [f for f in os.listdir(directory) if f.endswith('.txt')]
        
        for file_name in story_files:
            try:
                with open(os.path.join(directory, file_name), 'r') as file:
                    story = file.read()
                    
                prompt = f"""
                Story: {story}
                Question: {question}
                Search for the answer in the story.
                Give 0(Answer Not Found) or 1(Answer).
                Answer:"""
                
                chat_completion = self.client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="mixtral-8x7b-32768",
                    temperature=0.1,
                    max_tokens=500,
                    top_p=1,
                    stream=False
                )
                
                answer = chat_completion.choices[0].message.content
                if not answer.strip().startswith("0(Answer Not Found)"):
                    return answer
                    
            except Exception as e:
                print(f"Error processing {file_name}: {str(e)}")
                continue
        
        return "No answer found in any file."

# Flask Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_audio():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
            
        file = request.files['audio']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
            
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Initialize Whisper model
        device = torch.device("cpu")
        processor = WhisperProcessor.from_pretrained("MyAiModel", local_files_only=True)
        model = WhisperForConditionalGeneration.from_pretrained("MyAiModel", local_files_only=True)
        model.to(device)
        
        # Process audio
        chunks = split_audio(filepath)
        kannada_texts = []
        
        for chunk in chunks:
            text = transcribe_audio(chunk, processor, model, device)
            if text:
                kannada_texts.append(text)
        
        kannada_text = " ".join(kannada_texts)
        
        # Translate to English
        english_text = query_nllb(kannada_text, "kan_Knda", "eng_Latn")
        if not english_text:
            return jsonify({'error': 'Translation failed'}), 500
            
        # Process with Groq
        qa_system = StoryQASystem()
        groq_answer = qa_system.find_answer(english_text)
        
        # Translate answer back to Kannada
        kannada_answer = query_nllb(groq_answer, "eng_Latn", "kan_Knda")
        
        # Clean up
        for chunk in chunks:
            if os.path.exists(chunk):
                os.remove(chunk)
        if os.path.exists(filepath):
            os.remove(filepath)
        gc.collect()
        
        return jsonify({
            'kannada_text': kannada_text,
            'english_text': english_text,
            'groq_answer_english': groq_answer,
            'groq_answer_kannada': kannada_answer
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
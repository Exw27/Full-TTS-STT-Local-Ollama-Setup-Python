import torch
import subprocess
import speech_recognition as sr
import numpy as np
import os
import re
import whisper
import ollama
from TTS.api import TTS
import time
import warnings
from vosk import Model, KaldiRecognizer
import json
import logging
import psutil
from pydub import AudioSegment
import tempfile
import asyncio
import signal
import configparser
import tkinter as tk

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='logs/app.log')

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.conv")
warnings.filterwarnings("ignore", category=UserWarning, module="TTS.vocoder.layers.pqmf")

# Load configuration
config = configparser.ConfigParser()
config.read('config.ini')

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load models
whisper_model = whisper.load_model(config['Models']['whisper_model'])
vosk_model = Model(config['Models']['vosk_model'])
tts = TTS(config['Models']['tts_model']).to(device)

# Initialize other components
recognizer = sr.Recognizer()
ollama_api = config['Ollama']['api']
ollama_model = config['Ollama']['model']

# Initialize Tkinter
root = tk.Tk()
root.title("AI Assistant")
canvas = tk.Canvas(root, width=200, height=200)
canvas.pack()
circle = canvas.create_oval(50, 50, 150, 150, fill="blue")

# Define a function to update the UI circle
def update_circle(color, scale):
    canvas.itemconfig(circle, fill=color)
    canvas.coords(circle, 100-scale, 100-scale, 100+scale, 100+scale)
    root.update_idletasks()

# Define a function to pulsate the circle
async def pulsate_circle(color):
    scale = 50
    while True:
        for i in range(10, 50, 2):
            update_circle(color, i)
            await asyncio.sleep(0.05)
        for i in range(50, 10, -2):
            update_circle(color, i)
            await asyncio.sleep(0.05)

# Define a function to normalize audio data
def normalize_audio(audio_data):
    return audio_data.astype(np.float32) / 32767.0

# Define a function to transcribe audio and generate text
async def transcribe_audio(audio):
    try:
        raw_audio = audio.get_raw_data(convert_rate=16000, convert_width=2)
        audio_np = np.frombuffer(raw_audio, dtype=np.int16)
        audio_np_normalized = normalize_audio(audio_np)
        result = whisper_model.transcribe(audio_np_normalized)
        text = result["text"].strip()
        logging.info(f"Transcribed text: {text}")

        if not text:
            logging.info("Transcribed text is empty. Skipping Ollama API call.")
            return

        text = remove_emojis(text)
        response_text = await get_ollama_response(text)
        response_text = remove_emojis(response_text)
        logging.info(f"Ollama response: {response_text}")
        await speak_response(response_text)
    except Exception as e:
        logging.error(f"Error during transcription and response: {e}")

# Define a function to get response from Ollama API
async def get_ollama_response(text):
    try:
        stream = ollama.chat(
            model=ollama_model,
            messages=[{'role': 'user', 'content': text}],
            stream=True,
        )
        response_text = "".join(chunk['message']['content'] for chunk in stream)
        return response_text
    except Exception as e:
        logging.error(f"Error during Ollama API call: {e}")
        return ""

# Define a function to remove emojis from text
def remove_emojis(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

# Define a function to speak the response using TTS
async def speak_response(response_text):
    try:
        pulsate_task = asyncio.create_task(pulsate_circle("red"))
        tts.tts_to_file(text=response_text, file_path="output.wav")
        play_generated_speech("output.wav")
        logging.info(f"Response text: {response_text}")
        pulsate_task.cancel()
        update_circle("blue", 50)
    except Exception as e:
        logging.error(f"Error during TTS and response speaking: {e}")

# Define a function to play the generated speech
def play_generated_speech(file_path):
    try:
        subprocess.run(['aplay', file_path], stderr=subprocess.DEVNULL)
    except Exception as e:
        logging.error(f"Error during playing generated speech: {e}")


# Define a function to perform noise cancellation
def noise_cancellation(audio):
    try:
        raw_audio = audio.get_raw_data(convert_rate=16000, convert_width=2)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(raw_audio)
            temp_audio_path = f.name
        audio_segment = AudioSegment.from_file(temp_audio_path, format="wav")
        noise_sample = audio_segment[:1000]
        reduced_noise_audio = audio_segment.reduce_noise(noise_sample, verbose=False)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            reduced_noise_audio.export(f.name, format="wav")
            cleaned_audio_path = f.name
        with sr.AudioFile(cleaned_audio_path) as source:
            cleaned_audio = recognizer.record(source)
        return cleaned_audio
    except Exception as e:
        logging.error(f"Error during noise cancellation: {e}")
        return audio

# Define a function to listen for audio until a wake word is detected
async def listen_for_audio(wake_words=None):
    if wake_words is None:
        wake_words = [
            "hi", "hello", "you", "hey", "hey there",
            "good morning", "good afternoon", "good evening", "hi there",
            "excuse me", "can you", "could you", "would you", "please", "okay"
        ]
    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=2)  # Adjust for ambient noise for 2 seconds
            logging.info("Listening for wake word...")
            while True:
                pulsate_task = asyncio.create_task(pulsate_circle("blue"))
                audio = recognizer.listen(source, timeout=5)  # Adjust timeout as needed
                try:
                    raw_audio = audio.get_raw_data(convert_rate=16000, convert_width=2)
                    rec = KaldiRecognizer(vosk_model, 16000)
                    rec.AcceptWaveform(raw_audio)
                    result = json.loads(rec.Result())
                    spoken_text = result.get('text', '').lower()
                    logging.info(f"Spoken: {spoken_text}")
                    if any(word in spoken_text for word in wake_words):
                        logging.info("Wake word detected. Listening...")
                        pulsate_task.cancel()
                        update_circle("green", 50)
                        for char in 'T':
                            keyboard.press(char)
                            keyboard.release(char)
                            time.sleep(0.1)
                        break
                    else:
                        logging.info("No wake word detected. Listening again...")
                        pulsate_task.cancel()
                        update_circle("blue", 50)
                except sr.UnknownValueError:
                    logging.info("Could not understand audio. Listening again...")
                    pulsate_task.cancel()
                    update_circle("blue", 50)
        return audio
    except Exception as e:
        logging.error(f"Error during audio listening: {e}")

# Function to monitor memory usage
def monitor_memory(threshold=80):
    memory_usage = psutil.virtual_memory().percent
    if memory_usage > threshold:
        logging.warning(f"High memory usage detected: {memory_usage}%")

# Define a function to handle graceful shutdown
def handle_shutdown(signal, frame):
    logging.info("Shutdown signal received. Exiting gracefully...")
    exit(0)

# Function to run a health check
def health_check():
    try:
        if not torch.cuda.is_available() and device == "cuda":
            logging.error("CUDA device not available.")
        if not os.path.exists(config['Models']['vosk_model']):
            logging.error("Vosk model not found.")
    except Exception as e:
        logging.error(f"Health check error: {e}")

# Main loop
async def main_loop():
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)
    while True:
        try:
            monitor_memory()
            health_check()
            audio = await listen_for_audio()
            audio = noise_cancellation(audio)
            await transcribe_audio(audio)
        except Exception as e:
            logging.error(f"Unexpected error in main loop: {e}")

# Function to periodically run the asyncio event loop within Tkinter
def run_asyncio_event_loop():
    loop = asyncio.get_event_loop()
    loop.create_task(main_loop())
    loop.run_forever()

if __name__ == "__main__":
    root.after(100, run_asyncio_event_loop)
    root.mainloop()

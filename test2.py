import os
import tkinter as tk
from tkinter import messagebox, scrolledtext
import speech_recognition as sr
import numpy as np
import soundfile as sf
from python_speech_features import mfcc
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import threading
import pyaudio

# Initialize recognizer
recognizer = sr.Recognizer()

# Dictionary for storing users' data and model
users_data = {}
X_train = []
y_train = []
model = None

# Standard text for reading in three languages
texts = {
    'uz': "Bu ovoz tanish testi. Iltimos, bu matnni yodda saqlang.",
    'ru': "Это тест на распознавание голоса. Пожалуйста, запомните этот текст.",
    'en': "This is a voice recognition test. Please remember this text."
}

current_lang = 'en'  # Default language


# Function to record and save audio
def record_audio(file_path, live_wave=False):
    with sr.Microphone() as source:
        info_label.config(text="Recording... Please speak.")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

        if live_wave:
            # Start live waveform plotting in another thread
            plot_live_waveform(audio)

        with open(file_path, "wb") as f:
            f.write(audio.get_wav_data())
        info_label.config(text="Recording complete!")


# Function to extract features from the audio file
def extract_features(audio_file):
    data, samplerate = sf.read(audio_file)
    mfcc_features = mfcc(data, samplerate)
    return np.mean(mfcc_features, axis=0)


# Function to train the model with all users' data
def train_model():
    global model, X_train, y_train
    if not users_data:
        messagebox.showerror("Training Error", "Please record at least one user first!")
        return

    # Train the RandomForest model on all users' data
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    result_text.insert(tk.END, "Training complete for all users.\n")


# Function to test the model with new audio
def test_model():
    global model
    if not model:
        messagebox.showerror("Model Error", "Please train the system first!")
        return

    # Record the test voice
    record_audio("test_voice.wav", live_wave=True)
    test_features = extract_features("test_voice.wav")

    # Predict the user based on the voice features
    predicted_label = model.predict(test_features.reshape(1, -1))[0]
    recognized_user = None
    for user_name, user_data in users_data.items():
        if user_data["label"] == predicted_label:
            recognized_user = user_name
            break

    if recognized_user:
        result_text.insert(tk.END, f"Voice recognized as: {recognized_user}\n")
        result_text.config(fg='green')
    else:
        result_text.insert(tk.END, "Voice not recognized.\n")
        result_text.config(fg='red')

    plot_waveform("test_voice.wav")


# Function to plot waveform from an audio file
def plot_waveform(audio_file):
    data, samplerate = sf.read(audio_file)
    fig, ax = plt.subplots()
    ax.plot(data)
    ax.set_title('Waveform of User Voice')
    ax.set_xlabel('Samples')
    ax.set_ylabel('Amplitude')

    # Clear previous plots
    for widget in canvas_frame.winfo_children():
        widget.destroy()

    canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
    canvas.draw()
    canvas.get_tk_widget().pack()


# Function to dynamically plot live waveform
def plot_live_waveform(audio):
    def audio_stream_thread():
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)
        fig, ax = plt.subplots()
        while stream.is_active():
            data = np.frombuffer(stream.read(1024), dtype=np.int16)
            ax.clear()
            ax.plot(data)
            plt.pause(0.01)

        stream.stop_stream()
        stream.close()
        p.terminate()

    threading.Thread(target=audio_stream_thread).start()


# Function to record voice for a user
def record_voice():
    global X_train, y_train
    user_name = name_entry.get()
    if user_name:
        record_audio(f"{user_name}_voice.wav")
        user_features = extract_features(f"{user_name}_voice.wav")

        # Assign a unique label to each user
        user_label = len(users_data) + 1
        users_data[user_name] = {"features": user_features, "label": user_label}

        # Add to training set
        X_train.append(user_features)
        y_train.append(user_label)

        result_text.insert(tk.END, f"Voice recorded for user: {user_name}\n")
    else:
        messagebox.showerror("Input Error", "Please enter your name!")


# Language change handler
def change_language(lang):
    global current_lang
    current_lang = lang
    standard_text_label.config(text=texts[lang])


# Set up the GUI window
window = tk.Tk()
window.title("Voice Recognition System")
window.geometry("700x500")

# Variables for storing audio files and models
user_audio_file = 'user_voice.wav'
test_audio_file = 'test_voice.wav'

# GUI Elements
tk.Label(window, text="Enter your name:").pack()
name_entry = tk.Entry(window)
name_entry.pack()

# Info label and text box for results
info_label = tk.Label(window, text="")
info_label.pack()

result_text = scrolledtext.ScrolledText(window, height=5)
result_text.pack()

# Standard text label
standard_text_label = tk.Label(window, text=texts[current_lang])
standard_text_label.pack(pady=10)

# Language selection buttons
lang_frame = tk.Frame(window)
lang_frame.pack(pady=5)
tk.Button(lang_frame, text="Uzbek", command=lambda: change_language('uz')).pack(side=tk.LEFT)
tk.Button(lang_frame, text="Russian", command=lambda: change_language('ru')).pack(side=tk.LEFT)
tk.Button(lang_frame, text="English", command=lambda: change_language('en')).pack(side=tk.LEFT)

# Action buttons
button_frame = tk.Frame(window)
button_frame.pack(pady=10)
tk.Button(button_frame, text="Record Voice", command=record_voice).pack(side=tk.LEFT, padx=10)
tk.Button(button_frame, text="Train", command=train_model).pack(side=tk.LEFT, padx=10)
tk.Button(button_frame, text="Test Voice", command=test_model).pack(side=tk.LEFT, padx=10)

# Frame for the waveform plot
canvas_frame = tk.Frame(window)
canvas_frame.pack(fill=tk.BOTH, expand=True)

# Start the GUI event loop
window.mainloop()

import os
import tkinter as tk
from tkinter import messagebox, filedialog, scrolledtext
import speech_recognition
import numpy as np
import soundfile as sf
from python_speech_features import mfcc
from sklearn.mixture import GaussianMixture
import pickle
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import threading
# import pyaudio

# Initialize recognizer
recognizer = speech_recognition.Recognizer()

# Dictionary for storing users' data and models
users_data = {}
gmm_models = {}  # GMM models for each user
model_file = 'gmm_models.pkl'  # Default model file for GMM

# Function to record and save audio
def record_audio(file_path, live_wave=True):
    with speech_recognition.Microphone() as source:
        info_label.config(text="Yozilmoqda... Iltimos, gapiring.")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

        if live_wave:
            plot_live_waveform(audio)

        with open(file_path, "wb") as f:
            f.write(audio.get_wav_data())
        info_label.config(text="Yozish tugadi!")

# Function to extract features from the audio file
def extract_features(audio_file):
    data, samplerate = sf.read(audio_file)
    mfcc_features = mfcc(data, samplerate)
    return mfcc_features  # Return the full MFCC feature array

# Function to save the GMM models to a file
def save_gmm_models():
    global gmm_models
    if gmm_models:
        try:
            with open(model_file, 'wb') as f:
                pickle.dump(gmm_models, f)
            result_text.insert(tk.END, "Model muvaffaqiyatli saqlandi.\n")
        except Exception as e:
            result_text.insert(tk.END, f"Modelni saqlashda xato: {e}\n")
    else:
        messagebox.showerror("Saqlashda xato", "Hech qanday model saqlanmadi!")

# Function to load GMM models from file
def load_gmm_models():
    global gmm_models
    try:
        file_path = filedialog.askopenfilename(filetypes=[("Pickle files", "*.pkl")])
        if file_path:
            with open(file_path, 'rb') as f:
                gmm_models = pickle.load(f)
            result_text.insert(tk.END, f"Model {file_path} dan muvaffaqiyatli yuklandi.\n")
        else:
            result_text.insert(tk.END, "Hech qanday fayl tanlanmadi.\n")
    except Exception as e:
        result_text.insert(tk.END, f"Modelni yuklashda xato: {e}\n")

# Function to train GMM for each user
def train_gmm():
    global gmm_models
    if not users_data:
        messagebox.showerror("Xatolik", "Iltimos, avval foydalanuvchi yozib oling!")
        return

    for user_name, data in users_data.items():
        features = np.vstack(data["features"])  # Stack all features for the user
        if features.shape[0] < 2:
            messagebox.showerror("Trenirovka xatosi", f"{user_name} uchun yetarli ma'lumot yo'q. Iltimos, ko'proq ovoz yozing.")
            return
        gmm = GaussianMixture(n_components=8, covariance_type='diag', n_init=3)
        gmm.fit(features)
        gmm_models[user_name] = gmm

    result_text.insert(tk.END, "GMM modeli barcha foydalanuvchilar uchun trenirovka qilindi.\n")

# Function to test the model with new audio
def test_gmm():
    global gmm_models
    if not gmm_models:
        messagebox.showerror("Xatolik", "Iltimos, avval modelni yuklang yoki trenirovka qiling!")
        return

    record_audio("test_voice.wav", live_wave=True)
    test_features = extract_features("test_voice.wav")  # Now an array of features

    max_score = -float('inf')
    recognized_user = None

    # Test against each user's GMM model
    for user_name, gmm in gmm_models.items():
        scores = np.array(gmm.score(test_features))
        total_score = scores.sum()
        if total_score > max_score:
            max_score = total_score
            recognized_user = user_name

    if recognized_user:
        result_text.insert(tk.END, f"Ovoz tanildi: {recognized_user}\n")
        result_text.config(fg='green')
    else:
        result_text.insert(tk.END, "Ovoz tanilmadi.\n")
        result_text.config(fg='red')

    plot_waveform("test_voice.wav")

# Function to plot waveform from an audio file
def plot_waveform(audio_file):
    data, samplerate = sf.read(audio_file)
    fig, ax = plt.subplots()
    ax.plot(data)
    ax.set_title('Ovozning to‘lqini')
    ax.set_xlabel('Namuna')
    ax.set_ylabel('Amplituda')

    for widget in canvas_frame.winfo_children():
        widget.destroy()

    canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Function to dynamically plot live waveform
def plot_live_waveform(audio):
    # This is a placeholder for the live waveform functionality
    pass  # Implement if needed

# Function to record voice for a user
def record_voice():
    global users_data
    user_name = name_entry.get()
    if user_name:
        record_audio(f"{user_name}_voice.wav")
        user_features = extract_features(f"{user_name}_voice.wav")
        if user_name not in users_data:
            users_data[user_name] = {"features": []}
        users_data[user_name]["features"].append(user_features)

        result_text.insert(tk.END, f"Ovoz yozib olindi: {user_name}\n")
    else:
        messagebox.showerror("Xatolik", "Iltimos, ismingizni kiriting!")

# Set up the GUI window
window = tk.Tk()
window.title("Ovoz Tanish Tizimi")
window.geometry("800x600")

# Make elements dynamically resizable
window.rowconfigure(0, weight=1)
window.columnconfigure(0, weight=1)

# GUI Elements
tk.Label(window, text="Ismingizni kiriting:", font=('Arial', 14)).pack(pady=10)
name_entry = tk.Entry(window, font=('Arial', 14))
name_entry.pack(pady=10)

# Info label and text box for results
info_label = tk.Label(window, text="", font=('Arial', 14))
info_label.pack(pady=10)

result_text = scrolledtext.ScrolledText(window, height=6, font=('Arial', 14))
result_text.pack(pady=10, fill=tk.BOTH, expand=True)

# Action buttons in Uzbek
button_frame = tk.Frame(window)
button_frame.pack(pady=20, fill=tk.BOTH, expand=True)

tk.Button(button_frame, text="Ovoz yozish", command=record_voice, font=('Arial', 14)).pack(side=tk.LEFT, padx=10)
tk.Button(button_frame, text="GMM trenirovka", command=train_gmm, font=('Arial', 14)).pack(side=tk.LEFT, padx=10)
tk.Button(button_frame, text="Ovoz sinov", command=test_gmm, font=('Arial', 14)).pack(side=tk.LEFT, padx=10)
tk.Button(button_frame, text="Modelni saqlash", command=save_gmm_models, font=('Arial', 14)).pack(side=tk.LEFT, padx=10)
tk.Button(button_frame, text="Modelni yuklash", command=load_gmm_models, font=('Arial', 14)).pack(side=tk.LEFT, padx=10)

# Frame for the waveform plot
canvas_frame = tk.Frame(window)
canvas_frame.pack(fill=tk.BOTH, expand=True)

# Start the GUI event loop
window.mainloop()

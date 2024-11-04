import os
import tkinter as tk
from tkinter import messagebox, filedialog, scrolledtext
import speech_recognition as sr
import numpy as np
import soundfile as sf
from python_speech_features import mfcc
from sklearn.ensemble import RandomForestClassifier
import pickle
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
model_file = 'models.pkl'  # Default model file

# Standard text for reading in Uzbek
texts = {
    'uz': "Bu ovoz tanish testi. Iltimos, bu matnni yodda saqlang.",
}

# Function to record and save audio
def record_audio(file_path, live_wave=True):
    with sr.Microphone() as source:
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
    return np.mean(mfcc_features, axis=0)

# Function to save the model to a file
def save_model():
    global model, users_data, X_train, y_train
    if model:
        try:
            if os.path.exists(model_file):
                with open(model_file, 'rb') as f:
                    existing_data = pickle.load(f)
                    existing_data['X_train'].extend(X_train)
                    existing_data['y_train'].extend(y_train)
                    existing_data['users_data'].update(users_data)
                with open(model_file, 'wb') as f:
                    pickle.dump(existing_data, f)
            else:
                with open(model_file, 'wb') as f:
                    pickle.dump({'X_train': X_train, 'y_train': y_train, 'users_data': users_data}, f)
            result_text.insert(tk.END, "Model muvaffaqiyatli saqlandi.\n")
        except Exception as e:
            result_text.insert(tk.END, f"Modelni saqlashda xato: {e}\n")
    else:
        messagebox.showerror("Saqlashda xato", "Hech qanday model saqlanmadi!")

# Function to load model from file
def load_model():
    global model, users_data, X_train, y_train
    try:
        file_path = filedialog.askopenfilename(filetypes=[("Pickle files", "*.pkl")])
        if file_path:
            with open(file_path, 'rb') as f:
                loaded_data = pickle.load(f)
                X_train = loaded_data['X_train']
                y_train = loaded_data['y_train']
                users_data = loaded_data['users_data']

            model = RandomForestClassifier()
            model.fit(X_train, y_train)

            result_text.insert(tk.END, f"Model {file_path} dan muvaffaqiyatli yuklandi.\n")
        else:
            result_text.insert(tk.END, "Hech qanday fayl tanlanmadi.\n")
    except Exception as e:
        result_text.insert(tk.END, f"Modelni yuklashda xato: {e}\n")

# Function to train the model with all users' data
def train_model():
    global model, X_train, y_train
    if not users_data:
        messagebox.showerror("Xatolik", "Iltimos, avval foydalanuvchi yozib oling!")
        return

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    result_text.insert(tk.END, "Trenirovka tugallandi.\n")

# Function to test the model with new audio
def test_model():
    global model
    if not model:
        messagebox.showerror("Xatolik", "Iltimos, avval modelni yuklang yoki trenirovka qiling!")
        return

    record_audio("test_voice.wav", live_wave=True)
    test_features = extract_features("test_voice.wav")

    predicted_label = model.predict(test_features.reshape(1, -1))[0]
    recognized_user = None
    for user_name, user_data in users_data.items():
        if user_data["label"] == predicted_label:
            recognized_user = user_name
            break

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
    ax.set_ylabel('Amplitude')

    for widget in canvas_frame.winfo_children():
        widget.destroy()

    canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

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
        user_label = len(users_data) + 1
        users_data[user_name] = {"features": user_features, "label": user_label}

        X_train.append(user_features)
        y_train.append(user_label)

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
tk.Button(button_frame, text="Trenirovka", command=train_model, font=('Arial', 14)).pack(side=tk.LEFT, padx=10)
tk.Button(button_frame, text="Ovoz sinov", command=test_model, font=('Arial', 14)).pack(side=tk.LEFT, padx=10)
tk.Button(button_frame, text="Modelni saqlash", command=save_model, font=('Arial', 14)).pack(side=tk.LEFT, padx=10)
tk.Button(button_frame, text="Modelni yuklash", command=load_model, font=('Arial', 14)).pack(side=tk.LEFT, padx=10)

# Frame for the waveform plot
canvas_frame = tk.Frame(window)
canvas_frame.pack(fill=tk.BOTH, expand=True)

# Start the GUI event loop
window.mainloop()

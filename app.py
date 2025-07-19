import tkinter as tk
from tkinter import messagebox
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load model and vectorizer
with open("fake_news_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Predict Function
def detect_fake_news():
    user_input = text_input.get("1.0", tk.END).strip()
    if not user_input:
        messagebox.showwarning("Input Required", "Please enter a news article.")
        return

    transformed_text = vectorizer.transform([user_input])
    prediction = model.predict(transformed_text)[0]

    if prediction == "FAKE":
        result_label.config(text="❌ This news appears to be FAKE", fg="#e63946")
    else:
        result_label.config(text="✅ This news appears to be REAL", fg="#2a9d8f")

    animate_result(result_label)

# Typewriter Welcome Animation
def typewriter_effect(text, label, index=0):
    if index < len(text):
        label.config(text=label.cget("text") + text[index])
        app.after(50, lambda: typewriter_effect(text, label, index + 1))

# Simple animation for result fade-in
def animate_result(label):
    label.config(fg=label.cget("fg"))
    label.after(100, lambda: label.config(font=("Arial", 16, "bold")))

# App UI Setup
app = tk.Tk()
app.title("📰 Fake News Detector")
app.geometry("700x600")
app.configure(bg="#f4f4f8")

# Fonts
HEADER_FONT = ("Helvetica", 26, "bold")
TEXT_FONT = ("Arial", 12)
BUTTON_FONT = ("Arial", 14, "bold")

# Title
title = tk.Label(app, text="Fake News Detector", font=HEADER_FONT, bg="#f4f4f8", fg="#2c3e50")
title.pack(pady=(20, 5))

# Welcome message with typewriter animation
welcome_label = tk.Label(app, text="", font=("Arial", 14), bg="#f4f4f8", fg="#7f8c8d")
welcome_label.pack(pady=(0, 10))
typewriter_effect("👋 Welcome! Paste a news article below 👇", welcome_label)

# Text input box
text_input = tk.Text(app, height=10, width=70, font=TEXT_FONT, bd=2, relief=tk.GROOVE, wrap=tk.WORD)
text_input.pack(padx=20, pady=10)

# Suggestion tip
tip_label = tk.Label(app, text="💡 Tip: Copy real news from trusted websites or test with fake headlines", 
                     font=("Arial", 11, "italic"), bg="#f4f4f8", fg="#34495e")
tip_label.pack(pady=(5, 15))

# Predict button
predict_btn = tk.Button(app, text="🧠 Predict", command=detect_fake_news,
                        font=BUTTON_FONT, bg="#2c3e50", fg="white", padx=20, pady=10,
                        activebackground="#34495e", cursor="hand2", bd=0)
predict_btn.pack(pady=10)

# Result label
result_label = tk.Label(app, text="", font=("Arial", 16), bg="#f4f4f8")
result_label.pack(pady=20)

# Run the app
app.mainloop()

import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import pickle
import numpy as np

# Load vectoriser and model
with open('vectoriser.pickle', 'rb') as f:
    vectoriser = pickle.load(f)

with open('logisticRegression.pickle', 'rb') as f:
    model = pickle.load(f)

# Prediction function
def predict():
    input_text = text_entry.get("1.0", tk.END).strip()
    if not input_text:
        messagebox.showwarning("Input Error", "Please enter some text.")
        return

    X = vectoriser.transform([input_text])
    pred = model.predict(X)

    if hasattr(pred[0], '__iter__'):
        pred = np.round(pred).astype(int)

    label = "Positive 😊" if pred[0] == 1 else "Negative 😞"
    result_var.set(label)

# GUI Setup
root = tk.Tk()
root.title("Sentiment Analysis App")
root.geometry("600x400")
root.configure(bg="#f4f4f4")

# Heading
title = tk.Label(root, text="Sentiment Analyzer", font=("Helvetica", 20, "bold"), bg="#f4f4f4", fg="#333")
title.pack(pady=15)

# Input box
frame = tk.Frame(root, bg="#f4f4f4")
frame.pack(pady=10)

label = tk.Label(frame, text="Enter your text:", font=("Helvetica", 12), bg="#f4f4f4")
label.pack(anchor="w")

text_entry = tk.Text(frame, height=6, width=60, font=("Helvetica", 11))
text_entry.pack()

# Predict button
predict_button = tk.Button(
    root,
    text="Predict Sentiment",
    command=predict,
    font=("Helvetica", 12, "bold"),
    bg="#007acc",
    fg="white",
    activebackground="#005f99",
    padx=10,
    pady=5
)
predict_button.pack(pady=15)

# Result label
result_var = tk.StringVar()
result_label = tk.Label(root, textvariable=result_var, font=("Helvetica", 16), bg="#f4f4f4", fg="#222")
result_label.pack()

# Footer
footer = tk.Label(root, text="Built by Ahmed Saad", font=("Helvetica", 10), bg="#f4f4f4", fg="#888")
footer.pack(side="bottom", pady=10)

root.mainloop()

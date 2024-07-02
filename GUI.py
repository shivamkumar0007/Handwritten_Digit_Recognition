from tkinter import *
from tkinter import filedialog, messagebox
from PIL import ImageTk, Image
import cv2
import tensorflow as tf
import numpy as np

# Load the MNIST model
model = tf.keras.models.load_model('mnist.h5')
IMG_SIZE = 28

# Create the GUI
root = Tk()
root.title("Handwritten Digit Recognition")
root.geometry("300x200")

# Function to recognize the digit
def recognize_digit():
    try:
        image_file = filedialog.askopenfilename()
        img = cv2.imread(image_file)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized_img = cv2.resize(img_gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        new_img = tf.keras.utils.normalize(resized_img, axis=1)
        new_img = np.array(new_img).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
        prediction = model.predict(new_img)
        predicted_digit = np.argmax(prediction)
        label.config(text="Predicted Digit: " + str(predicted_digit))
    except cv2.error:
        messagebox.showerror("Error", "Image not found. Please provide the correct path.")

# Create the GUI components
label = Label(root, text="Select an image to recognize the digit", pady=10)
label.pack()

button = Button(root, text="Browse", command=recognize_digit)
button.pack()

root.mainloop()

import cv2
from ultralytics import YOLO
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

def load_model(model_path):
    return YOLO(model_path)


def read_image(image_path):
    return cv2.imread(image_path)


def predict_objects(model, image):
    results = model.predict(source=image)
    boxes = results[0].boxes.xyxy
    scores = results[0].boxes.conf
    class_ids = results[0].boxes.cls
    return boxes, scores, class_ids


def crop_objects(image, boxes):
    cropped_images = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        cropped_image = image[y1:y2, x1:x2]
        cropped_images.append(cropped_image)
    return cropped_images

def save_cropped_images(cropped_images, save_folder):
    os.makedirs(save_folder, exist_ok=True)
    for i, cropped_image in enumerate(cropped_images):
        save_path = os.path.join(save_folder, f'cropped_image_{i}.jpg')
        cv2.imwrite(save_path, cropped_image)


def main(model_path, image_path, save_folder):
    model = load_model(model_path)
    image = read_image(image_path)
    boxes, _, _ = predict_objects(model, image)
    if len(boxes) > 0:
        cropped_images = crop_objects(image, boxes)
        save_cropped_images(cropped_images, save_folder)
        messagebox.showinfo("Success", f"Cropped images saved to {save_folder}")
    else:
        messagebox.showinfo("No objects detected", "No objects detected in the image.")


def browse_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if file_path:
        image_path.set(file_path)


def browse_save_folder():
    folder_path = filedialog.askdirectory()
    if folder_path:
        save_folder.set(folder_path)

def start_detection():
    if not image_path.get() or not save_folder.get():
        messagebox.showerror("Error", "Please select an image and a save folder.")
        return

    main('vails.pt', image_path.get(), save_folder.get())


root = tk.Tk()
root.title("Object Detection GUI")


image_path = tk.StringVar()
save_folder = tk.StringVar()


ttk.Label(root, text="Select Image:").grid(row=0, column=0, padx=10, pady=10, sticky="e")
ttk.Entry(root, textvariable=image_path, width=50).grid(row=0, column=1, padx=10, pady=10)
ttk.Button(root, text="Browse", command=browse_image).grid(row=0, column=2, padx=10, pady=10)

ttk.Label(root, text="Select Save Folder:").grid(row=1, column=0, padx=10, pady=10, sticky="e")
ttk.Entry(root, textvariable=save_folder, width=50).grid(row=1, column=1, padx=10, pady=10)
ttk.Button(root, text="Browse", command=browse_save_folder).grid(row=1, column=2, padx=10, pady=10)

ttk.Button(root, text="Start Detection", command=start_detection).grid(row=2, columnspan=3, pady=20)


root.mainloop()
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
from ultralytics import YOLO


def load_model(model_path):
    return YOLO(model_path)


def predict_objects(model, image):
    results = model.predict(source=image)
    boxes = results[0].boxes.xyxy
    return boxes


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


def detect_from_image(image_path, save_folder):
    model_path = 'vails.pt'
    model = load_model(model_path)
    
    if image_path.lower().endswith(('.jpg', '.jpeg', '.png')):
        image = cv2.imread(image_path)
        boxes = predict_objects(model, image)
        
        if len(boxes) > 0:
            cropped_images = crop_objects(image, boxes)
            save_cropped_images(cropped_images, save_folder)
            messagebox.showinfo("Success", f"Cropped images saved to {save_folder}")
        else:
            messagebox.showinfo("No objects detected", "No objects detected in the image.")
    else:
        messagebox.showerror("Error", "Unsupported file format. Please select a valid image file.")


def detect_from_camera(save_folder):
    model_path = 'vails.pt'
    model = load_model(model_path)
    
    savings_folder = save_folder
    os.makedirs(savings_folder, exist_ok=True)

    cap = cv2.VideoCapture(0)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        boxes = predict_objects(model, frame)

        if len(boxes) > 0:
            x1, y1, x2, y2 = map(int, boxes[0])

            cropped_image = frame[y1:y2, x1:x2]

            save_path = os.path.join(savings_folder, f'cropped_image_{frame_count}.jpg')
            cv2.imwrite(save_path, cropped_image)
            frame_count += 1

        cv2.imshow('Live Stream', frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def browse_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if file_path:
        image_entry.delete(0, tk.END)
        image_entry.insert(0, file_path)


def browse_save_folder():
    folder_path = filedialog.askdirectory()
    if folder_path:
        save_folder_entry.delete(0, tk.END)
        save_folder_entry.insert(0, folder_path)


def start_detection():
    if image_radio_var.get() == 0:
        if not image_entry.get() or not save_folder_entry.get():
            messagebox.showerror("Error", "Please select an image and a save folder.")
            return
        detect_from_image(image_entry.get(), save_folder_entry.get())
    elif image_radio_var.get() == 1:
        if not save_folder_entry.get():
            messagebox.showerror("Error", "Please select a save folder.")
            return
        detect_from_camera(save_folder_entry.get())


root = tk.Tk()
root.title("Object Detection GUI")

image_radio_var = tk.IntVar()
image_radio_var.set(0)

ttk.Radiobutton(root, text="Select Image", variable=image_radio_var, value=0).grid(row=0, column=0, padx=10, pady=10)
image_entry = ttk.Entry(root, width=50)
image_entry.grid(row=0, column=1, padx=10, pady=10)
ttk.Button(root, text="Browse", command=browse_image).grid(row=0, column=2, padx=10, pady=10)

ttk.Radiobutton(root, text="Use Camera", variable=image_radio_var, value=1).grid(row=1, column=0, padx=10, pady=10)
save_folder_entry = ttk.Entry(root, width=50)
save_folder_entry.grid(row=1, column=1, padx=10, pady=10)
ttk.Button(root, text="Browse", command=browse_save_folder).grid(row=1, column=2, padx=10, pady=10)

start_button = ttk.Button(root, text="Start Detection", command=start_detection)
start_button.grid(row=2, columnspan=3, pady=20)

root.mainloop()

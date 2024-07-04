import cv2
from ultralytics import YOLO
import os


model = YOLO(r'C:\Users\RAJ MOHNANI\OneDrive\Desktop\vails project\vails.pt')


savings_folder = 'C:/Users/RAJ MOHNANI/OneDrive/Desktop/vails project/savings'
os.makedirs(savings_folder, exist_ok=True)


cap = cv2.VideoCapture(0)

frame_count = 0

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to capture image")
        break

 
    results = model.predict(source=frame)

   
    boxes = results[0].boxes.xyxy
    scores = results[0].boxes.conf
    class_ids = results[0].boxes.cls

    for i, score in enumerate(scores):
        if score >= 0.1:  
            x1, y1, x2, y2 = map(int, boxes[i])
            
            
            cropped_image = frame[y1:y2, x1:x2]

            
            save_path = os.path.join(savings_folder, f'cropped_image_{frame_count}.jpg')
            cv2.imwrite(save_path, cropped_image)
            frame_count += 1

   
    cv2.imshow('Live Stream', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

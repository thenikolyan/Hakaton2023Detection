import os
import pandas as pd
import datetime as dt

import cv2
from ultralytics import YOLO


modelset1 = YOLO(rf'best.pt')
#modelset2 = YOLO('best.pt')

#video_path = rf"C:\Users\nikol\Desktop\train\weapon\test.mp4"
video_path = rf"test.mp4"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

df = []
frame_number = 0
names = ["person", "knife", "cell phone", "scissors", "gun", "rifle"]

if not os.path.exists("runs/detect/predict"):
    os.makedirs("runs/detect/predict")

while cap.isOpened():
    success, frame = cap.read()
    frame_number += 1

    if success:
        results = modelset1.track(frame, persist=True, conf=0.0, save=False, save_txt=False)
        # print("Boxes:", results[0].boxes)
        # print("Probs:", results[0].probs)
        if results and results[0].boxes and results[0].boxes.data.shape[0] > 0 and False:
            
            img_with_boxes = results[0].orig_img.copy()
            current_time_in_seconds = frame_number / fps


            for box, conf_value, class_id in zip(results[0].boxes.xyxy, results[0].boxes.conf, results[0].boxes.cls):
                x1, y1, x2, y2 = [int(coord) for coord in box]
                label = f"Class {results[0].names[int(class_id.item())]}: {conf_value:.2f}"
                cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 0, 0), 2)
                cv2.putText(img_with_boxes, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

                file_prefix = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
                cv2.imwrite(f"runs/detect/predict/{file_prefix}_boxed.jpg", img_with_boxes)


            with open(f"runs/detect/predict/{file_prefix}.txt", "w") as f:
                for box, conf_value, class_id in zip(results[0].boxes.xyxy, results[0].boxes.conf, results[0].boxes.cls):
                    x1, y1, x2, y2 = box
                    f.write(f"{results[0].names[int(class_id.item())]} {x1} {y1} {x2} {y2} {conf_value.item()}\n")

                    df.append({'Time': current_time_in_seconds, 'class': results[0].names[int(class_id.item())], 'confidence': conf_value.item(), 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})

        annotated_frame = results[0].plot()

        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break
pd.DataFrame(df).to_excel('test3.xlsx', index=False)
cap.release()
cv2.destroyAllWindows()

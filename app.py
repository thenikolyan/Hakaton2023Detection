from flask import Flask, request, render_template, Response
import os
from werkzeug.utils import secure_filename

import datetime as dt
import pandas as pd

import cv2
from ultralytics import YOLO


app = Flask(__name__)

UPLOAD_FOLDER = rf'files\upload'  # Укажите путь к папке для загрузки
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}  # Допустимые форматы видео
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        bool_option = 'bool_option' in request.form
        float_value = request.form.get('float_value', type=float)
        if 'file' in request.files:
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                return render_template('index.html', filename=filename, bool_option=bool_option, float_value=float_value)

    return render_template('index.html')

def gen(filename, bo=False, fo=.5):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    model = YOLO(rf'yolo\best.pt')
    cap = cv2.VideoCapture(file_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    df = []
    frame_number = 0
    names = ["person", "knife", "cell phone", "scissors", "gun", "rifle"]
    dir_name = dt.datetime.now().strftime('%Y%m%d_%H%M%S')

    if not os.path.exists('output'):
                    os.makedirs('output')

    if not os.path.exists(rf'output\{dir_name}'):
        os.makedirs(rf'output\{dir_name}')
    
    if not os.path.exists(rf'output\{dir_name}\images'):
            os.makedirs(rf'output\{dir_name}\images')
            
    if not os.path.exists(rf'output\{dir_name}\labels'):
            os.makedirs(rf'output\{dir_name}\labels')

    while cap.isOpened():
        success, frame = cap.read()
        frame_number += 1

        if success:
            results = model.track(frame, persist=True, conf=fo, save=False, save_txt=False)
            if results and results[0].boxes and results[0].boxes.data.shape[0] > 0 and bo:
                
                img_with_boxes = results[0].orig_img.copy()
                current_time_in_seconds = frame_number / fps

                for box, conf_value, class_id in zip(results[0].boxes.xyxy, results[0].boxes.conf, results[0].boxes.cls):
                    x1, y1, x2, y2 = [int(coord) for coord in box]
                    label = f"Class {results[0].names[int(class_id.item())]}: {conf_value:.2f}"
                    cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 0, 0), 2)
                    cv2.putText(img_with_boxes, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

                    file_prefix = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
                    cv2.imwrite(rf"output\{dir_name}\images\{file_prefix}_boxed.jpg", img_with_boxes)


                with open(rf"output\{dir_name}\labels\{file_prefix}.txt", "w") as f:
                    for box, conf_value, class_id in zip(results[0].boxes.xyxy, results[0].boxes.conf, results[0].boxes.cls):
                        x1, y1, x2, y2 = box
                        f.write(f"{results[0].names[int(class_id.item())]} {x1} {y1} {x2} {y2} {conf_value.item()}\n")

                        df.append({'Time': current_time_in_seconds, 'class': results[0].names[int(class_id.item())], 'confidence': conf_value.item(), 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})

            annotated_frame = results[0].plot()

            #cv2.imshow("YOLOv8 Tracking", annotated_frame)
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    pd.DataFrame(df).to_excel(rf'output\{dir_name}\report.xlsx', index=False)
    cap.release()
    cv2.destroyAllWindows()

@app.route('/video_feed/<filename>')
def video_feed(filename):

    bo = request.args.get('bo', type=bool)
    fo = request.args.get('fo', type=float)
    return Response(gen(filename, bo, fo), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)

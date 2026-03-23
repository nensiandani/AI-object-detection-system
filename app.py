from flask import Flask, render_template, request, Response
import os
import cv2
from ultralytics import YOLO

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# YOLO મોડલ લોડ કરો
model = YOLO("yolov8n.pt") 

def generate_frames(source):
    # જો source 'webcam' હોય તો 0, બાકી ફાઈલ પાથ
    camera_source = 0 if source == 'webcam' else source
    cap = cv2.VideoCapture(camera_source)
    
    if not cap.isOpened():
        return None

    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # persist=True થી ઓબ્જેક્ટ ટ્રેકિંગ થશે, જેથી એકનો એક ઓબ્જેક્ટ ફરી કાઉન્ટ ન થાય
            results = model.track(frame, persist=True)
            annotated_frame = results[0].plot()
            
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    cap.release()

@app.route("/", methods=["GET", "POST"])
def index():
    input_img = None
    result_img = None
    counts = {}
    mode = None
    video_file = None

    if request.method == "POST":
        mode = request.form.get("mode")
        file = request.files.get("file")

        if mode == "webcam":
            return render_template("index.html", mode=mode)

        if file and file.filename != "":
            filename = file.filename
            path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(path)
            
            if mode == "image":
                input_img = filename
                results = model(path)
                res_plotted = results[0].plot()
                result_filename = "res_" + filename
                result_path = os.path.join(UPLOAD_FOLDER, result_filename)
                cv2.imwrite(result_path, res_plotted)
                result_img = result_filename

                # Perfect Count Logic
                for box in results[0].boxes:
                    label = model.names[int(box.cls[0])]
                    counts[label] = counts.get(label, 0) + 1
            
            elif mode == "video":
                video_file = filename
                # વિડિયોની પહેલી ફ્રેમ પરથી ઇનિશિયલ કાઉન્ટ
                results = model(path)
                for box in results[0].boxes:
                    label = model.names[int(box.cls[0])]
                    counts[label] = counts.get(label, 0) + 1

    return render_template(
        "index.html",
        input_img=input_img,
        result_img=result_img,
        counts=counts,
        mode=mode,
        video_file=video_file
    )

@app.route('/video_feed/<filename>')
def video_feed(filename):
    if filename == 'webcam':
        path = 'webcam'
    else:
        path = os.path.join(UPLOAD_FOLDER, filename)
        
    frames = generate_frames(path)
    if frames is None:
        return "Camera not found", 404
        
    return Response(frames, mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
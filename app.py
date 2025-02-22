from flask import Flask, render_template, Response, request, jsonify, send_file
import cv2
import numpy as np
from model import get_base_model
from utils import preprocess_fer
import tensorflow as tf
import base64
import os

# Flask application setup
app = Flask(__name__, template_folder='templates')

# Load emotion detection model
IMG_SHAPE = (100, 100, 3)
emotion_model = get_base_model(IMG_SHAPE)
emotion_model.add(tf.keras.layers.Dense(7, activation='softmax', name="softmax"))
#emotion_model.load_weights('model/FERplus_1228-2223.h5')
emotion_model.load_weights('model/FERplus_0124-1040_weights.h5')
emotions = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=8,  # 5'ten 8'e çıkardık
                minSize=(50, 50),  # 30x30'dan 50x50'ye çıkardık
            )

            for (x, y, w, h) in faces:
                # Yüz alanını al ve analiz et
                roi = frame[y:y + h, x:x + w]
                roi_resized = cv2.resize(roi, (100, 100))
                roi_preprocessed = preprocess_fer(np.expand_dims(roi_resized, axis=0))

                preds = emotion_model.predict(roi_preprocessed)[0]
                emotion_probability = np.max(preds)
                
                # Sadece yüksek olasılıklı tahminleri göster
                if emotion_probability > 0.2:  # Eşik değeri ekledik
                    label_index = np.argmax(preds)
                    label = emotions[label_index]

                    # Yazı için arka plan dikdörtgeni
                    text = f"{label} ({emotion_probability:.2f})"
                    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                    cv2.rectangle(frame, 
                                (x, y - text_height - 20), 
                                (x + text_width + 10, y - 5), 
                                (0, 0, 0), 
                                -1)  # Siyah arka plan

                    # Yazıyı ekle
                    cv2.putText(frame, 
                                text,
                                (x + 5, y - 10),  # Konumu ayarla
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.9,  # Font boyutu
                                (255, 255, 255),  # Beyaz yazı
                                2)  # Kalınlık

                    # Yüz çerçevesi
                    cv2.rectangle(frame, 
                                (x, y), 
                                (x + w, y + h),
                                (0, 255, 0),  # Yeşil çerçeve
                                2)


            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


def generate_frames_video(video_path):
    cap = cv2.VideoCapture(video_path)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=8,
                minSize=(50, 50),
                maxSize=(800, 800)  # Maximum boyut sınırı ekledik
            )

            for (x, y, w, h) in faces:
                roi = frame[y:y + h, x:x + w]
                roi_resized = cv2.resize(roi, (100, 100))
                roi_preprocessed = preprocess_fer(np.expand_dims(roi_resized, axis=0))

                preds = emotion_model.predict(roi_preprocessed)[0]
                emotion_probability = np.max(preds)
                
                if emotion_probability > 0.5:  # Eşik değeri ekledik
                    label_index = np.argmax(preds)
                    label = emotions[label_index]

                    text = f"{label} ({emotion_probability:.2f})"
                    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(frame, 
                                  (x, y - text_height - 20), 
                                  (x + text_width + 10, y - 5), 
                                  (0, 0, 0), 
                                  -1)  # Siyah arka plan

                    cv2.putText(frame, 
                                text,
                                (x + 5, y - 10),  # Konumu ayarla
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,  # Font boyutu
                                (255, 255, 255),  # Beyaz yazı
                                2)  # Kalınlık

                    cv2.rectangle(frame, 
                                  (x, y), 
                                  (x + w, y + h),
                                  (0, 255, 0),  # Yeşil çerçeve
                                  2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/image-detect', methods=['GET', 'POST'])
def image_detect():
    if request.method == 'GET':
        return render_template('image_detect.html')

    # Get uploaded image
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=8, minSize=(60, 60))

    results = []
    for (x, y, w, h) in faces:
        roi = img[y:y + h, x:x + w]
        roi_resized = cv2.resize(roi, (100, 100))
        roi_preprocessed = preprocess_fer(np.expand_dims(roi_resized, axis=0))

        preds = emotion_model.predict(roi_preprocessed)[0]
        emotion_probability = np.max(preds)
        label_index = np.argmax(preds)
        label = emotions[label_index]

        results.append({
            'label': label,
            'probability': float(emotion_probability),
            'box': [int(x), int(y), int(w), int(h)]
        })

        text = f"{label} ({emotion_probability:.2f})"
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)

        cv2.rectangle(img, 
                          (x, y - text_height - 20), 
                          (x + text_width + 10, y - 5), 
                          (0, 0, 0), 
                          -1)

        cv2.putText(img, 
                    text,
                    (x + 5, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2)

        cv2.rectangle(img, 
                        (x, y), 
                        (x + w, y + h),
                        (0, 255, 0),
                          2)

    _, buffer = cv2.imencode('.jpg', img)
    img_str = base64.b64encode(buffer).decode('utf-8')

    return render_template('image_detect.html', results=results, image_data=img_str)


@app.route('/camera-detect')
def camera_detect():
    return render_template('camera_detect.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed_video/<path:video_path>')
def video_feed_video(video_path):
    return Response(generate_frames_video(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video-detect', methods=['GET', 'POST'])
def video_detect():
    if request.method == 'GET':
        return render_template('video_detect.html')

    # Get uploaded video
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    file = request.files['video']
    video_path = 'uploaded_video.mp4'
    file.save(video_path)

    return render_template('video_detect.html', video_path=video_path)




# if __name__ == "__main__":
#     app.run(debug=True)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render'ın atadığı portu kullan
    app.run(host="0.0.0.0", port=port, debug=True)
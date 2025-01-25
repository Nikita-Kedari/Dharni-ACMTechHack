from flask import Flask, request, jsonify, render_template
import os
import io
import base64
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)


model = YOLO("best.pt")  

def draw_bounding_boxes(image, results):
    """
    Draw bounding boxes on the image based on the model output.
    """
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  
            conf = float(box.conf[0])  
            cls = int(box.cls[0])  

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = f"Class {cls}: {conf:.2f}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image

def image_to_base64(image):
    buffered = io.BytesIO()
    result_image = Image.fromarray(image)
    result_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

@app.route('/')
def index():
    return render_template('./index2.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    gps = request.form.get('coordinates')

    if file:
       
        filepath = os.path.join('uploads', file.filename)
        file.save(filepath)

        image = Image.open(filepath)
        image = np.array(image)

        results = model.predict(source=image, save=False, conf=0.25)

        image_with_bboxes = draw_bounding_boxes(image, results)

        img_str = image_to_base64(image_with_bboxes)

        response = {
            'image': img_str,
            'severity': 'Critical',
            'location': gps or 'Unknown',
            'suggestion': 'Immediate repair required.'
        }
        return jsonify(response)

    return jsonify({'error': 'No file uploaded'}), 400

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)

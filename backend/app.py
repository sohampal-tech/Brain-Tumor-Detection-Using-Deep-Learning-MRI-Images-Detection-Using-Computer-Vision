import os
import json
import zipfile
import tempfile
import h5py
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, send_from_directory, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Flatten, Dropout, Dense
from tensorflow.keras.applications import VGG16

# 1. Folder paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TEMPLATE_DIR = os.path.join(BASE_DIR, 'frontend', 'templates')
STATIC_DIR = os.path.join(BASE_DIR, 'frontend', 'static')
MODEL_PATH = os.path.join(BASE_DIR, 'mri_brain_tumor_model.keras')
CLASS_NAMES_PATH = os.path.join(BASE_DIR, 'class_names.json')

# Uploads are saved OUTSIDE the project folder so VS Code Live Server
# does not see a new file and refresh the page.
UPLOAD_FOLDER = os.path.join(tempfile.gettempdir(), 'mri_uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    return response


# 2. Load class names
if os.path.exists(CLASS_NAMES_PATH):
    with open(CLASS_NAMES_PATH, 'r') as f:
        class_labels = json.load(f)
else:
    class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

# 3. Build the VGG16 model architecture
IMAGE_SIZE = 128
base_model = VGG16(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False, weights=None)

model = Sequential([
    Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
    base_model,
    Flatten(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(class_labels), activation='softmax')
])

# 4. Load weights from the .keras file
with zipfile.ZipFile(MODEL_PATH, 'r') as z:
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
        tmp.write(z.read('model.weights.h5'))
        tmp_path = tmp.name

try:
    with h5py.File(tmp_path, 'r') as f:
        vgg_convs = [l for l in base_model.layers if len(l.weights) > 0]
        for i, layer in enumerate(vgg_convs):
            grp_name = 'conv2d' if i == 0 else f'conv2d_{i}'
            grp = f['layers']['functional']['layers'][grp_name]['vars']
            layer.set_weights([grp['0'][:], grp['1'][:]])

        dense_layers = [l for l in model.layers if isinstance(l, Dense)]
        for i, layer in enumerate(dense_layers):
            grp_name = 'dense' if i == 0 else f'dense_{i}'
            grp = f['layers'][grp_name]['vars']
            layer.set_weights([grp['0'][:], grp['1'][:]])
finally:
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

print("mri_brain_tumor_model.keras loaded successfully!")


# 5. Prediction function
def predict_tumor(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)
    predicted_class_index = int(np.argmax(predictions, axis=1)[0])
    confidence_score = float(np.max(predictions, axis=1)[0])

    label = class_labels[predicted_class_index]
    if label.lower() == 'notumor':
        return "No Tumor", confidence_score
    return f"Tumor: {label}", confidence_score


# 6. Routes
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    try:
        file = request.files.get('file')
        if not file or not file.filename:
            return jsonify({'error': 'No file uploaded'}), 400

        filename = secure_filename(file.filename) or 'upload.jpg'
        file_location = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_location)

        result, confidence = predict_tumor(file_location)
        print(f"Predicted: {result} ({confidence * 100:.2f}%)")

        return jsonify({
            'result': result,
            'confidence': f"{confidence * 100:.2f}%"
        })
    except Exception as e:
        print("Prediction Error:", str(e))
        return jsonify({'error': str(e)}), 500


@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    # debug=False so the model loads only once and the server never auto-restarts
    app.run(host='127.0.0.1', port=5000, debug=False)
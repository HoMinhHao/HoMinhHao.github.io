from flask import Flask, request, render_template, jsonify
import os
from werkzeug.utils import secure_filename
import time
from tqdm import tqdm
from deepface import DeepFace
from pathlib import Path
import numpy as np
app = Flask(__name__)

# Configure folders for uploads and known faces
UPLOAD_FOLDER = 'uploaded_images'
KNOWN_FACES_FOLDER = 'template_faces'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload and known faces folders exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(KNOWN_FACES_FOLDER):
    os.makedirs(KNOWN_FACES_FOLDER)

# Allowed file extensions for image uploads
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def upload_form():
    return render_template('index.html')

@app.route('/upload-image', methods=['POST'])
def upload_image():
    # Check if the request contains the file part
    if 'image' not in request.files:
        return jsonify({'error': 'No image part in the request'}), 400

    file = request.files['image']

    # If the user does not select a file
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        face_objs = DeepFace.find(
            img_path = filepath,
            model_name = "ArcFace",
            detector_backend = "skip",
            align = True,
            db_path = KNOWN_FACES_FOLDER,
            threshold=0.6
        )

        face_objs=np.array(face_objs)

        if len(face_objs[0])>0:
            # Define the path
            file_path = Path(face_objs[0][0][0])
 
            # Extract the filename without extension
            filename = file_path.stem
            return jsonify(filename), 200
        else:
            return jsonify("Stranger"), 401

    return jsonify({'error': 'File type not allowed'}), 400

if __name__ == '__main__':
    app.run(debug=True)

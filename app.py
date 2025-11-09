import os
import requests
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
from PIL import Image

app = Flask(__name__)

# --- KONFIGURASI MODEL BARU DENGAN GOOGLE DRIVE ---

# 1. Definisi Model
MODEL_FILENAME = 'serangga_cnn_model_v1.h5'
# GANTI DENGAN ID FILE YANG SUDAH ANDA DAPATKAN DARI GOOGLE DRIVE
GOOGLE_DRIVE_FILE_ID = "/d/1UlihhYIUJfX83ZOxlEHCJzsc7H0CB7rS/view?usp=sharing" 

# 2. Path
IMAGE_SIZE = (150, 150) 
CLASS_NAMES = ['aphids', 'armyworm', 'beetle', 'bollworm', 'grasshopper', 
               'mites', 'mosquito', 'sawfly', 'stem_borer'] 
NUM_CLASSES = len(CLASS_NAMES)


def download_file_from_google_drive(id, destination):
    """Mengunduh file dari Google Drive menggunakan file ID."""
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

def get_confirm_token(response):
    """Mendapatkan token konfirmasi jika file terlalu besar."""
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

# --- MUAT MODEL SAAT SERVER DIMULAI ---
model = None
print(f"Memulai pengunduhan model ({MODEL_FILENAME})...")

try:
    # 1. Unduh model dari Google Drive
    download_file_from_google_drive(GOOGLE_DRIVE_FILE_ID, MODEL_FILENAME)
    print("Pengunduhan selesai. Memuat model...")
    
    # 2. Muat Model dari file lokal yang baru diunduh
    model = tf.keras.models.load_model(MODEL_FILENAME)
    print("Model Deep Learning berhasil dimuat.")
    
except Exception as e:
    print(f"Gagal mengunduh atau memuat model: {e}")
    # Jika gagal, model tetap None, dan endpoint /predict akan error
# ----------------------------------------
    
def preprocesses_image(img):
    """Fungsi untuk preprocessing gambar sebelum diumpankan ke model"""
    img = img.resize(IMAGE_SIZE)
    img_array = img_to_array(img)
    img_array = img_array/255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/')
def home():
    """Melayani file index.html dari folder templates"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint untuk menerima upload gambar dan melakukan prediksi"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error' : 'No selected file'}), 400
    
    if file:
        try:
            img = Image.open(file.stream)
            
            processed_img = preprocesses_image(img)
            predictions = model.predict(processed_img)
            predicted_index = np.argmax(predictions[0])
            predicted_class = CLASS_NAMES[predicted_index]
            confidence = float(predictions[0][predicted_index])
            
            response = {
                'status': 'success',
                'prediction': predicted_class,
                'confidence' : f"{confidence*100:.2f}"
            }
            return jsonify(response)
        except Exception as e:
            return jsonify({'error': f'Predicted failed: {str(e)}'}), 500

if __name__ == '__main__':

    app.run(debug=True, host='0.0.0.0', port=5000)

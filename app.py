import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from PIL import Image
import json
import pandas as pd
import os

# Load model
model = load_model('model/skin_disease_model.keras')

# Load class indices
with open('model/class_indices.json', 'r') as f:
    class_indices = json.load(f)

# Urutkan label berdasarkan indeks
class_labels = [label for label, _ in sorted(class_indices.items(), key=lambda x: x[1])]

# Fungsi prediksi
def predict_image(image):
    image = image.resize((150, 150))  # Sesuaikan dengan ukuran input model
    img_array = img_to_array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    return class_labels[np.argmax(predictions)], predictions

# Antarmuka Streamlit
st.title("Aplikasi Diagnosis Penyakit Kulit")
st.write("Unggah gambar kulit untuk mendapatkan prediksi penyakit kulit.")

# Input gambar
uploaded_file = st.file_uploader("Pilih file gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = load_img(uploaded_file)
    st.image(image, caption="Gambar yang diunggah", use_column_width=True)
    st.write("Memproses gambar...")

    # Prediksi
    label, prediction = predict_image(image)
    st.write(f"Prediksi penyakit kulit: **{label}**")

    # Visualisasi probabilitas
    df = pd.DataFrame({
        'Kategori': class_labels,
        'Probabilitas': prediction[0]
    })
    st.bar_chart(df.set_index('Kategori'))

    # Detail penyakit
    with st.expander("Detail Penyakit"):
        if label == 'Normal':
            st.write("Penyakit kulit normal, tidak ada kelainan yang terdeteksi.")
        elif label == 'Kanker':
            st.write("Melanoma adalah jenis kanker kulit yang sering kali dimulai sebagai bintik hitam atau tahi lalat. Periksa ke dokter segera.")
        elif label == 'Arsenik':
            st.write("Keracunan arsenik dapat menyebabkan perubahan warna kulit, pembengkakan, dan munculnya bercak putih pada kulit.")
        elif label == 'Psoriasis':
            st.write("Psoriasis adalah gangguan autoimun yang menyebabkan kulit menjadi merah, bersisik, dan terkelupas.")
        elif label == 'Eksim':
            st.write("Eksim adalah kondisi kulit yang menyebabkan gatal, kemerahan, dan ruam yang dapat menyebar ke seluruh tubuh.")

    # Solusi
    with st.expander("Solusi dan Penanganan"):
        if label == 'Kanker':
            st.write("Segera konsultasikan dengan dokter untuk perawatan lebih lanjut, seperti biopsi atau operasi.")
        elif label == 'Arsenik':
            st.write("Berhenti terpapar arsenik dan konsultasikan dengan dokter untuk pengobatan lebih lanjut.")
        elif label == 'Psoriasis':
            st.write("Psoriasis bisa dikendalikan dengan terapi topikal, obat-obatan, atau fototerapi. Konsultasikan dengan dokter kulit.")
        elif label == 'Eksim':
            st.write("Eksim dapat dikendalikan dengan pelembab kulit, antihistamin, dan perawatan medis yang tepat.")
        else:
            st.write("Untuk penyakit kulit normal, tidak diperlukan pengobatan lebih lanjut.")

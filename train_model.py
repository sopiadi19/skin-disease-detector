import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping
import os
import json

# Path ke direktori dataset
dataset_path = 'data/'

# Preprocessing gambar dengan augmentasi
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,  # 20% data untuk validasi
    rotation_range=20,     # Rotasi gambar hingga 20 derajat
    width_shift_range=0.2, # Perpindahan horizontal hingga 20%
    height_shift_range=0.2,# Perpindahan vertikal hingga 20%
    shear_range=0.2,       # Distorsi shearing
    zoom_range=0.2,        # Zoom gambar hingga 20%
    horizontal_flip=True   # Membalikkan gambar secara horizontal
)

# Generator untuk data training
train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

# Generator untuk data validasi
val_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Menampilkan urutan label
print("Urutan label berdasarkan direktori dataset:")
print(train_generator.class_indices)

# Simpan urutan label ke file JSON
if not os.path.exists('model/'):
    os.makedirs('model/')
with open('model/class_indices.json', 'w') as f:
    json.dump(train_generator.class_indices, f)

# Membangun model CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(5, activation='softmax')  # 5 kelas
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callback EarlyStopping untuk menghentikan pelatihan jika tidak ada peningkatan
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Melatih model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=30,
    callbacks=[early_stopping]
)

# Menyimpan model
model.save('model/skin_disease_model.keras')

# Simpan riwayat pelatihan ke file JSON
with open('model/history.json', 'w') as f:
    json.dump(history.history, f)

# Evaluasi akhir pada data validasi
val_loss, val_accuracy = model.evaluate(val_generator)
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy}")

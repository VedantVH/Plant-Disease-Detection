# train_model.py

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import json
import os

# 📁 Update this to your PlantVillage folder path
DATASET_PATH = "C:/Users/VIGHNESH/Desktop/Vighnesh_related/internship 5th sem/internship_5_project-old/archive (1)/PlantVillage"  # Must contain class folders
#C:\Users\VIGHNESH\Desktop\Vighnesh_related\internship 5th sem\internship_5_project-old\archive (1)\PlantVillage

# 🔁 Data Preparation
img_size = 224
batch_size = 32

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# 🔡 Save class index mapping
with open("class_indices.json", "w") as f:
    json.dump(train.class_indices, f)

# 🧠 MobileNetV2 Model
base_model = MobileNetV2(
    input_shape=(img_size, img_size, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # Freeze base model

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(train.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 🏋️ Training
model.fit(train, validation_data=val, epochs=5)

# 💾 Save model
model.save("plant_disease_model.h5")

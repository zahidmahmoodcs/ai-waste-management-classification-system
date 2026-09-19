import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image

model = tf.keras.models.load_model(
    "best_model.keras",
    custom_objects={"preprocess_input": preprocess_input}
)

classes = ['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash']  # FIX THIS

img = Image.open("static/uploads/cardboard1.jpg").convert('RGB')
img = img.resize((224,224))

img = np.array(img)
img = preprocess_input(img)
img = np.expand_dims(img, axis=0)

pred = model.predict(img)

print("Raw:", pred)
print("Class:", classes[np.argmax(pred)])
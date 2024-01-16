import cv2
import numpy as np
from tensorflow.keras.preprocessing import image

def load_image(img_path):
    img = image.load_img(img_path, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0

def extract_cnn_features(model, img_path):
    img_array = load_image(img_path)
    features = model.predict(img_array)
    return features.flatten()
import cv2
import numpy as np
from tensorflow.keras.models import load_model


# loading the model
model = load_model('model.h5', custom_objects=None, compile=True)

# Loading the image from file path
filepath = input('Enter the path of the picture you would like to classifiy: ')
picture = cv2.imread(filepath)

# Resizing image to fit neural network
img_dim = (256, 256)
picture = cv2.resize(picture, img_dim)
print(picture.shape)
print(picture)

# evaluating and guessing the image
guess = model.predict(picture)
print(guess)
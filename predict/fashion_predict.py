from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras as keras
import numpy as np
import matplotlib.pyplot as plt
import sys
from PIL import Image

print(tf.__version__)

print("Loading {0} for prediction".format(sys.argv[1]))

# Load a trained model from the current working directory
model = keras.models.load_model("trained_model.h5")

# Get the input image size
_, height, width, depth = model.layers[0].input_shape

# Load the image to predict using the size of the model's first input layer
# The fashion MNIST dataset uses images of size (28 x 28); grayscale
predict_image = tf.keras.preprocessing.image.load_img(sys.argv[1], target_size=[height, width], color_mode='grayscale')

predict_array = tf.keras.preprocessing.image.img_to_array(predict_image)

print("Image shape is: {0}".format(predict_array.shape))

# Save the predict_image converted to input standards for this model
# Large images scaled to small shapes usually morph into blobs and may be classified as a bag
tf.keras.preprocessing.image.save_img('small.png', predict_array)

# Scale entries to between 0 and 1
predict_array = predict_array / 255.0

# Remembering the shape of our training and test image, use NumPy to insert the image 
# into the zeroith index for use in the predicting this one image
img = (np.expand_dims(predict_array, 0))

# Use the trained model to classify our image
predictions_single = model.predict(img)

# Convert scientific notation to float. This only affects the display of the numbers
np.set_printoptions(formatter={'float_kind':'{:f}'.format})

# Multiply each array item by 100
predictions_single = predictions_single * 100

# Note that the item at index = 9 is very close to 1.0 and the other array values are not
print(predictions_single)

# These are the names of each of our ten (10) classes of clothing
class_names = ['Tee-Shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Set up matplotlib to produce a plot of our prediction
y_pos = np.arange(len(class_names))

# predictions_single is currently an array within an array. We flatten it to remove the outer array
predictions_single = predictions_single.flatten()

plt.rcdefaults()
plt.bar(y_pos, predictions_single, align='center', alpha=0.5)
plt.xticks(y_pos, class_names, rotation=45)
plt.ylabel('Probability')
plt.title('Image Classification')

# Save a png of our prediction to disk; check your current working directory
plt.savefig("prediction")

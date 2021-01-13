
import tensorflow as tf
import numpy as np


NAMED_CATEGORIES = ["Wave", "Peacesign", "Three", "Paper", "Fist", "Horns", "One", "Four", "Ok", "Thumb", "Pinky"]


model = tf.keras.models.load_model("saved_model")
img = tf.keras.preprocessing.image.load_img("10-color.png", target_size=(480, 640), color_mode = "grayscale")

img.show()

img_array = tf.keras.preprocessing.image.img_to_array(img)
img = tf.expand_dims(img_array, 0)

prediction = model.predict(img);
score = tf.nn.softmax(prediction)

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(NAMED_CATEGORIES[np.argmax(score)], 100 * np.max(score))
)
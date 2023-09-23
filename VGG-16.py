import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from PIL import Image

model = keras.applications.VGG16()

img = Image.open('images/car.jpg')
plt.imshow(img)

img = np.array(img)
x = keras.applications.vgg16.preprocess_input(img)

print(x.shape)

x = np.expand_dims(x, axis=0)
res = model.predict(x)

print(np.argmax(res))

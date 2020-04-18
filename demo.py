import cv2
import numpy as np

import model

from tensorflow.python.keras.applications.vgg16 import preprocess_input

img = preprocess_input(cv2.imread('./tmp/test.jpg'))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (512, 512))

model_dlv3 = model.Deeplabv3()
predicted = model_dlv3.predict(img[np.newaxis, ...])

person_score = predicted[0, :, :, 15]
back_score = predicted[0, :, :, 0]

mask = (person_score > back_score).astype("uint8") * 255

cv2.imwrite("./tmp/person_musk.jpg", mask)

from tensorflow.keras.models import load_model
import cv2, numpy as np


myModel = load_model('myModel.h5')

def predict(img):
    # img = cv2.imread(img)
    nimg = np.array(img)
    ocvim = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
    img = cv2.resize(ocvim, (256, 256))
    out = myModel.predict(np.expand_dims(img / 255, axis = 0))

    if out >= 0.5:
        return 0, out[0][0]
    return 1, out[0][0]

# print(predict('test_1.png'))
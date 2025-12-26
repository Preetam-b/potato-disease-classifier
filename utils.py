import numpy as np
import tensorflow as tf
from PIL import Image
IMG_SIZE= 256
def preprocess_image(image:Image.Image):
    """
    Preprocess image for Mobilenet_v2 model.
    Input :PIL.Image
    Output : Numpy array of shape(1,256,256,3)
    """
    image=image.convert("RGB")
    image=image.resize((IMG_SIZE,IMG_SIZE))
    img_array=np.array(image)
    img_array=tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array=np.expand_dims(img_array,axis=0)

    return img_array
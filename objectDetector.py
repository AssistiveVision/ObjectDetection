# load yolov3 model and perform object detection
from numpy import expand_dims
import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

# load and prepare an image (Output: scaled pixel data, original width and height of the image)
def load_image_pixels(filename, shape):
    # load the image to get its shape
    image = load_img(filename)
    width, height = image.size
    # load the image with the required size
    image = load_img(filename, target_size=shape)
    # convert to numpy array
    image = img_to_array(image)
    # scale pixel values to [0, 1]
    image = image.astype('float32')
    image /= 255.0
    # add a dimension so that we have one sample
    image = expand_dims(image, 0)
    return image, width, height

# Main
# load yolov3 model
model = load_model('model.h5')
# The model expects inputs to be color images with the square shape of 416×416 pixels.

# define the expected input shape for the model
input_w, input_h = 416, 416
# define our new photo
photo_filename = 'zebra.jpg'
# load and prepare image
image, image_w, image_h = load_image_pixels(photo_filename, (input_w, input_h))

# make prediction
yhat = model.predict(image)
# summarize the shape of the list of arrays

# [(1, 13, 13, 255), (1, 26, 26, 255), (1, 52, 52, 255)]
# returns a list of three NumPy arrays, the shape of which is displayed as output.
# These arrays predict both the bounding boxes and class labels but are encoded. They must be interpreted.
print([a.shape for a in yhat])
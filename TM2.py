import tensorflow.keras
print("tensorflow")
from PIL import Image, ImageOps
print("PIL")
import numpy as np
print("numpy")

import cv2
from sense_hat import SenseHat

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)
print("printoptions")
# Load the model
model = tensorflow.keras.models.load_model('/home/pi/Documents/Model/TFdetectionLE/keras_model.h5')
print("model")

sense = SenseHat()

cap = cv2.VideoCapture(0)
cap.set(3,224)
cap.set(4,224)
    
while True:
    b=True
    i=1
    
    for event in sense.stick.get_events():
        if event.direction == "middle" and event.action == "pressed":
            
            while b == True:
                ret,im = cap.read()
                flipVertical = cv2.flip(im, 1)
                cv2.imshow('video test',flipVertical)
                key = cv2.waitKey(10)
                cv2.imwrite('/home/pi/Pictures/img/img.png', flipVertical)
                
                for event in sense.stick.get_events():

                    if event.direction == "middle" and event.action == "pressed":
                        cap.release()
                        cv2.destroyAllWindows()
                        
                        # Create the array of the right shape to feed into the keras model
                        # The 'length' or number of images you can put into the array is
                        # determined by the first position in the shape tuple, in this case 1.
                        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
                        print("data")
                        # Replace this with the path to your image
                        image = Image.open('/home/pi/Pictures/img/img.png')
                        print("image.open")
                        #resize the image to a 224x224 with the same strategy as in TM2:
                        #resizing the image to be at least 224x224 and then cropping from the center
                        size = (224, 224)
                        image = ImageOps.fit(image, size, Image.ANTIALIAS)
                        print("imageOps")
                        #turn the image into a numpy array
                        image_array = np.asarray(image)
                        print("image_array")
                        # display the resized image
                        image.show()
                        print("image.show")
                        # Normalize the image
                        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
                        print("normalized_image_array")
                        # Load the image into the array
                        data[0] = normalized_image_array
                        print("data image")
                        # run the inference
                        prediction = model.predict(data)
                        print(prediction, "prediction")
                        
                        prediction = str(prediction)
                        prediction = prediction.replace(' ', ',')
                        list(prediction)
                        
                        print(prediction)
                        
                        maxi=max(prediction)
                        print(maxi)
                        
                        
                        b = False

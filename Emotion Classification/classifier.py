import tensorflow as tf
import cv2
import numpy as np

class Classifier:
    """
    This class creates emotion classifier.

    ...

    Attributes
    ----------
    model_path : String
        path of Tensorflow weights containing classifier model

    classes_path : String
        path of a .csv containing the classes by row

    Methods
    -------
    predict(image)
        Classifies an emotion

    """

    def __init__(self, model_path, classes_path):

        self.model = tf.keras.models.load_model(model_path)

        classes = {}

        csv_file_classes = open(classes_path, encoding='utf-8').readlines()[1:]
        for idx, line in enumerate(csv_file_classes):
            classes[idx] = line.strip()

        self.classes = classes

    def predict(self, image):

        """
         Classifies an emotion

        Args:
            image (np.array): image frame

        Return:
            (String): emotion

        """

        # Copies image
        img = image.copy()

        # Prepare image for classifier model input
        img = cv2.resize(img, (48, 48))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        
        # Gets the classification result 
        pred_proba = self.model.predict(img)
        prediction = (np.argmax(pred_proba))

        return self.classes[prediction]
import cv2
from detector import Detector
from classifier import Classifier

class Pipeline:
    """
    This class represents a Emotion Classification pipeline.

    ...

    Methods
    -------
    execute(frame)
        Process a image frame on entire pipeline.

    """

    def __init__(self):

        # Creates pipeline detector
        self.detector = Detector('yolov5_faces.pt')
        
        # Creates pipeline classifier
        self.classifier = Classifier('fer2013_resnet.h5', 'classes.csv')

    def execute(self, frame):

        """
        Detects and extracts features of all faces in image.

        Args:
            frame (np.array): image frame

        """

        # Detects all faces in frame and gets its bounding boxes
        bboxes = self.detector.detect(frame)

        for (x_min, x_max, y_min, y_max) in bboxes:

            # Gets RoI
            face_img = frame[y_min:y_max, x_min:x_max]

            # Predict emotion
            emotion = self.classifier.predict(face_img)

            # Draws detection bounding box on image
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 155), 2)

            # Draws detection, color, make and model
            text = "{}".format(emotion)
            cv2.putText(frame, text, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 0, 155), 2)
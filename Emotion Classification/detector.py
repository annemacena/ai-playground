import torch

class Detector:
    """
    This class creates a face detector.

    ...

    Attributes
    ----------
    yolo_path : String
        path of Torch weights containing classifier model

    confidence : Float
        detection confidence

    Methods
    -------
    detect(image)
        detects faces in an image

    """

    def __init__(self, yolo_path, confidence = 0.20):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=yolo_path)
        self.confidence = confidence

    def detect(self, image):

        """
        Detects faces in an image

        Args:
            image (np.array): image frame

        Return:
            (list): detections bounding boxes

        """

        # Model inference
        detection_results = self.model(image)

        # Gets detected objects data
        df = detection_results.pandas().xyxy[0]

        # Gets objects indexes
        objects_idx = df.index

        bounding_boxes = []

        for idx in objects_idx:

            # Gets detection confidence
            confidence = df['confidence'][idx]
            
            if confidence > self.confidence:

                # Gets detection bounding box
                x_min, x_max = int(df['xmin'][idx]), int(df['xmax'][idx])
                y_min, y_max = int(df['ymin'][idx]), int(df['ymax'][idx])

                # Inserts current bounding box in list
                bounding_boxes.append((x_min, x_max, y_min, y_max))

        return bounding_boxes

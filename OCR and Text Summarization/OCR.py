import pytesseract

class OCR:
    """
    This class represents all steps involving OCR

    ...

    Methods
    -------
    image_to_string(image)
        returns the result of a OCR run on the provided image to string

    """

    def __init__(self):
        self.lang = 'eng+por'

    def image_to_string(self, image):
        """
        Returns the result of a OCR run on the provided image to string

        Args:
            image (object or string): file path of the image to be processed

        Return:
            Returns the result of a OCR run on the provided image to string

        """

        return pytesseract.image_to_string(image, lang=self.lang)
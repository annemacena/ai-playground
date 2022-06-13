import argparse
from ast import Raise
import cv2
import numpy as np
import time
from PIL import Image
import os

from pipeline import Pipeline
from PathType import PathType

import mimetypes
mimetypes.init()

def get_filename_incremented(file_extension):
    """
    Increments a number in the filename to not overwrite it

    ...

    Attributes
    ----------
    file_extension : String
        file extension (.mp4, .jpg, etc)
    """
    
    count = 0
    while os.path.exists(f"results/out{count}{file_extension}"):
        count += 1
        
    return f"results/out{count}{file_extension}"

def run_video_classification(opt):
    """
    Runs pipeline based on video file

    ...

    Attributes
    ----------
    opt : dict
        given arguments
    """

    cap = cv2.VideoCapture(opt.input)

    if (cap.isOpened() == False): 
        raise Exception("Error opening video stream or file")

    max_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video = None
    FPS = 0.0
    processed_frames = 0
    
    pipeline = Pipeline()

    # Reads all video frames
    while(cap.isOpened()):

        ret, frame = cap.read() # current frame
        
        if ret == True:
            
            start_time = time.time()

            pipeline.execute(frame)

            FPS += 1.0 / (time.time() - start_time)
            processed_frames += 1.0
            
            # Computes and shows processing percentage
            percentage = (processed_frames * 100) / max_frames
            
            if percentage % 5 == 0:
                print(f"{int(percentage)}% of the video has been processed")

            # Shows video
            if opt.debug:
                cv2.namedWindow("Emotion Classification",cv2.WINDOW_NORMAL)
                cv2.imshow('Emotion Classification', frame)

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

            # Saves video
            if opt.save:
                if video is None:
                    output_filename = get_filename_incremented(".mp4")
                    video = cv2.VideoWriter(output_filename,
                                            cv2.VideoWriter_fourcc(*'mp4v'), 
                                            10, (frame.shape[1],frame.shape[0]))

                video.write(np.array(frame, dtype=np.uint8))

        else: 
            break
    
    # Stops video reader
    cap.release()
    cv2.destroyAllWindows()

    # Computes and shows FPS
    FPS = FPS / processed_frames
    print(f"FPS = {FPS}")

def run_image_classification(opt, file_extension):
    """
    Runs pipeline based on image file

    ...

    Attributes
    ----------
    opt : dict
        given arguments

    file_extension : String
        file extension (.png, .jpg, etc)
    """

    img_file = Image.open(opt.input).convert('RGB')
    img_array = np.asarray(img_file)
    
    pipeline = Pipeline()

    pipeline.execute(img_array)
    img_result = Image.fromarray(img_array)

    # Shows video
    if opt.debug:
        img_result.show(img_array)

    # Saves image
    if opt.save:
        output_filename = get_filename_incremented(file_extension)
        img_result.save(output_filename)
    
def main(opt, file_extension):

    mimestart = mimetypes.guess_type(opt.input)[0]
    mimestart = mimestart.split('/')[0]
    
    if(mimestart == "video"):
        run_video_classification(opt)
    elif(mimestart == "image"):
        run_image_classification(opt, file_extension)
    else:
        raise Exception("The input file is not a video or an image")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True,
                        help="Path of input", 
                        type = PathType(exists=True, type='file'))

    parser.add_argument('--save', action ='store_true', help='Save result')

    parser.add_argument('--debug', action ='store_true', help='Draw info to debug')

    opt = parser.parse_args()
    file_extension = os.path.splitext(opt.input)[1]
    
    main(opt, file_extension)  
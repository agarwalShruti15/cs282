import cv2
import numpy as np
from imutils import face_utils
import argparse
import imutils
import dlib
import cv2
import math

"""
Function
--------
video_to_frame

Extract individual frames from a video using cv2 library:
Reference: https://www.learnopencv.com/read-write-and-display-a-video-using-opencv-cpp-python/


Parameters
-----------
file_path : string
    The path where the video file is stored.



Returns
-------
frame : Numpy array
    Array of dimension (frame_width,frame_height,3), that contains all the frames in the video.


"""




def video_to_frame(file_path):

    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    cap = cv2.VideoCapture(file_path)

    # Check if camera opened successfully
    if (cap.isOpened()== False):
      print("Error opening video stream or file")

    # Read until video is completed
    frames=[] # Initialize list to store frames
    while(cap.isOpened()):
      #Capture frame-by-frame
        ret, frame = cap.read()


        if ret == True: # Add frame to list
            #frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

            # Break the loop when there are no more frames
        else:
            break

    # When everything done, release the video capture object
    cap.release()
    frames=np.array(frames)

    return frames

"""
Function
--------
summarize_expressions

Takes an array that has all the frames in a video and averages out 'expression_lenght' consecutive frames:
Reference: https://www.learnopencv.com/read-write-and-display-a-video-using-opencv-cpp-python/


Parameters
-----------
video : Numpy array
    The video to be transformed as an array containing frames.
expression_length: int
    The number of consecutive frames that will be averaged out to create an expression



Returns
-------
video : Numpy array
    Video in 'summarized' format where each frame is an expression (average of frames)


"""

def summarize_expressions(video,expression_length):

    current_frames=video.shape[0]
    new_frames=current_frames//expression_length
    new_video=[]
    i=0


    while i<current_frames:
        frame_i=[]
        for j in range(expression_length):
            if(i+j>=current_frames):
                break
            else:
                frame_i.append(video[i+j])
                print



        #frame_i=np.array(np.around(frame_i/expression_length),dtype=np.uint8)
        frame_i=np.array(frame_i)
        frame_i=np.array(np.mean(frame_i,axis=0),dtype=np.uint8) # This double wrapping is need for matplotlib to render image correctly
        new_video.append(frame_i)
        i+=j+1


    new_video=np.array(new_video)

    return new_video



"""
Function
--------
calculate_landmarks

Takes video array (frame by frame) of shape (no_frames,frame_width,frame_height,color_channels) and creates an array with
facial landmarks for each array.
Reference: https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/


Parameters
-----------
video : Numpy array
    video array (frame by frame) of shape (no_frames,frame_width,frame_height,color_channels)



Returns
-------
landmarks : Numpy array
    Array that contains (x,y) coordinates for the facial features in each frame of the video


"""


def calculate_landmarks(video):

    landmarks=[]

    for image in video:

        predictor_path="facial-landmarks/shape_predictor_68_face_landmarks.dat"
        # initialize dlib's face detector (HOG-based) and then create
        # the facial landmark predictor
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(predictor_path)

        # load the input image, resize it, and convert it to grayscale
        #image = cv2.imread(args["image"])

        image = imutils.resize(image, width=500)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale image
        rects = detector(gray, 1)
        shape=0

        # loop over the face detections
        for (i, rect) in enumerate(rects):
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            landmarks.append(shape)

    landmarks=np.array(landmarks)

    return landmarks


"""
Function
--------
gram_matrix

Takes landmarks array of shape (no_frames,frame_width,frame_height,color_channels) and calculates gram matrix for each
of the frame landmarks in the array

Parameters
-----------
landmarks : Numpy array
   array of shape (no_frames,2) with facial landmarks (x-y-coordinates).


Returns
-------
gram_array : Numpy array
    Array that contains gram matrix for facial landmarks in each frame of the video.


"""

def gram_matrix(landmarks):
    gram_array=[]
    for landmark in landmarks:
        gram_i=np.dot(landmark,landmark.T)
        gram_array.append(gram_i)

    gram_array=np.array(gram_array)


    return gram_array

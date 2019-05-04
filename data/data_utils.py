
import os
import argparse
import numpy as np
import cv2

"""creates an outfile and write the name of the youtube link 
 using the id which is the name of the file. """
def links_from_files(fldr_nm, outfile):

    #get all the mp4 files
    yt_id = [f[0:11] for f in os.listdir(fldr_nm) if f.endswith('.mp4')]
    yt_id = np.unique(yt_id)
    with open(outfile, 'w') as f:
        [f.write('https://www.youtube.com/watch?v={}\n'.format(yt_id[i])) for i in range(len(yt_id))]

# return youtube link time in seconds
def get_link_duration(txt_file):
    import youtube_dl

    ydl_opts = {
    'outtmpl': 'tmp/%(id)s.%(ext)s',
    'noplaylist': True,
    'quiet': True,
    'forceduration':True
    }
    time = 0 #seconds
    files = 0
    with open(txt_file, 'r') as f:
        url = f.readlines()
        for u in url:
            try:
                with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                    dictMeta = ydl.extract_info(u.split()[0], download=False)
                    time = time + dictMeta['duration']
                files = files+1
            except:
                continue

    return time, files

#path: Path to directory containing text files with information about dataset
def get_vid_stats(path):
    import codecs
    import pandas as pd
    #----VARIABLE AND CONSTANT INITIALIZATION
    fps=30 # Constant to detMermine Frames per second on each video

    # Define empty lists
    user_id=[]
    video_id=[]
    no_frames=[]
    file_path=[]
    duration=[]
    split=[]


    # Define dictionary to create DataFrame
    dic={'split':'','user_id':'','video_id':'','no_frames':'','file_path':'','duration':''}

    # ----EXTRACT FILES IN SUB-DIRECTORIES
    for subdir, dirs, files in os.walk(path):

        for file in files:

            if file.endswith(".csv"):

                # ----GET FILE INFORMATION FOR EACH FILE (CLIP)

                # Get file path and split into directories
                file_path_i=os.path.join(subdir, file)
                path_array=file_path_i.split('/')

                # Get directory names that correspond to utterance (clip), video and user
                video_id_i=path_array[-1][:-4] # -4 is to eliminate .txt from string
                user_id_i=path_array[-3]
                split_i=path_array[-4]

                #'''
                # For each clip get number of frames by counting lines in .txt that describes clip
                with codecs.open(file_path_i, 'r', encoding='utf-8',errors='ignore') as f:
                    for i, l in enumerate(f):
                        pass
                no_frames_i=i-1 #Subtracting the lines in the top of the file

                # Calculate duration
                duration_i=no_frames_i/fps

                #'''
                # ----UPDATE LISTS WITH FILE (CLIP) INFORMATION
                user_id.append(user_id_i)
                video_id.append(video_id_i)
                file_path.append(file_path_i)
                split.append(split_i)
                no_frames.append(no_frames_i)
                duration.append(duration_i)

    # ----UPDATE DICTIONARY WITH POPULATED LISTS (remove first row which is root folder)
    dic['user_id']=user_id
    dic['split']=split
    dic['video_id']=video_id
    dic['no_frames']=no_frames
    dic['file_path']=file_path
    dic['duration']=duration

    #----CREATE DATAFRAME USING DICTIONARY
    return pd.DataFrame(data=dic)


#path: Path to directory containing text files with information about dataset
def get_vid_stats_fake(path):
    import codecs
    import pandas as pd
    #----VARIABLE AND CONSTANT INITIALIZATION
    fps=30 # Constant to detMermine Frames per second on each video

    # Define empty lists
    user_id=[]
    video_id=[]
    no_frames=[]
    file_path=[]
    duration=[]
    split=[]


    # Define dictionary to create DataFrame
    dic={'split':'','user_id':'','video_id':'','no_frames':'','file_path':'','duration':''}

    # ----EXTRACT FILES IN SUB-DIRECTORIES
    for subdir, dirs, files in os.walk(path):

        for file in files:

            if file.endswith(".csv"):

                # ----GET FILE INFORMATION FOR EACH FILE (CLIP)

                # Get file path and split into directories
                file_path_i=os.path.join(subdir, file)
                path_array=file_path_i.split('/')

                # Get directory names that correspond to utterance (clip), video and user
                video_id_i=path_array[-1][:-4] # -4 is to eliminate .txt from string
                user_id_i=path_array[-2]
                split_i=path_array[-3]

                #'''
                # For each clip get number of frames by counting lines in .txt that describes clip
                with codecs.open(file_path_i, 'r', encoding='utf-8',errors='ignore') as f:
                    for i, l in enumerate(f):
                        pass
                no_frames_i=i-1 #Subtracting the lines in the top of the file

                # Calculate duration
                duration_i=no_frames_i/fps

                #'''
                # ----UPDATE LISTS WITH FILE (CLIP) INFORMATION
                user_id.append(user_id_i)
                video_id.append(video_id_i)
                file_path.append(file_path_i)
                split.append(split_i)
                no_frames.append(no_frames_i)
                duration.append(duration_i)

    # ----UPDATE DICTIONARY WITH POPULATED LISTS (remove first row which is root folder)
    dic['user_id']=user_id
    dic['split']=split
    dic['video_id']=video_id
    dic['no_frames']=no_frames
    dic['file_path']=file_path
    dic['duration']=duration

    #----CREATE DATAFRAME USING DICTIONARY
    return pd.DataFrame(data=dic)


#code to get the face in an RGB image
def get_face(image, verbose=False):

    #used to detect the face in the frames
    protoPath = os.path.sep.join(['face_detection_model', "deploy.prototxt"])
    modelPath = os.path.sep.join(['face_detection_model', "res10_300x300_ssd_iter_140000.caffemodel"])
    DETECTOR = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    face = None
    (h, w) = image.shape[:2]
    # construct a blob from the image
    imageBlob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0),
                                      swapRB=False, crop=False)
    # apply OpenCV's deep learning-based face detector to localize faces in the input image
    DETECTOR.setInput(imageBlob)
    detections = DETECTOR.forward()
    if len(detections) > 0: # ensure at least one face was found
        # we're making the assumption that each image has only ONE
        # face, so find the bounding box with the largest probability
        i = np.argmax(detections[0, 0, :, 2])
        if (detections[0, 0, i, 2] > 0.9):

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # extract the face ROI and grab the ROI dimensions
            face = image[startY:endY, startX:endX]
            if verbose:
                cv2.imshow("Frame", face)
                key = cv2.waitKey(1) & 0xFF

            if face.shape[0] < 10 and face.shape[1] < 10:
                face = None

    return face

#get all the frames in the video, preprocess as the following:
#1 extract the rgb frame
#2 convert to grayscale
#3 reduce the size
#4 reduce the temporal frequency
def get_example_face(path, nframes = 1, face_size=(300, 300)):

    #the number of frames we will get from this video
    vidcap = cv2.VideoCapture(path) # read the mp4
    in_frame_cnt = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
    assert (nframes<=in_frame_cnt)

    ids = np.linspace(0, in_frame_cnt, nframes)
    success, image = vidcap.read() # extract the frames
    out_faces = np.zeros((len(ids), *face_size, image.shape[2]))
    i = 0
    while success:

        face = get_face(image) # get the face rectangle
        if face is not None:

            # reduce the frame size to frame_size (sz_x X sz_y X frames)
            out_faces[i, :, :, :] = cv2.cvtColor(cv2.resize(face, dsize=face_size, interpolation=cv2.INTER_CUBIC)
                                                         , cv2.COLOR_BGR2RGB)
            i += 1

        if i >= nframes:
            break
        success, image = vidcap.read() # extract the frames

    return out_faces




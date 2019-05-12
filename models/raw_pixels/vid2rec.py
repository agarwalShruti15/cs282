
from video2tfrecord_test3 import convert_videos_to_tfrecord
import time



people=['bernie','biden','hillary','justin','may','michelle','modi','obama','pelosi','putin','trump','warren']
#people=['modi']

for p in people:

    start=time.time()

    #---TRAIN--
    print('Processing TRAIN '+p)
    source='/home/ubuntu/fakebusters/data/train/'+p+'/real'
    dest='/home/ubuntu/fakebusters/data/train/'+p
    convert_videos_to_tfrecord(source, dest, 2000, 32, "*.mp4")

    '''
    #---VAL
    print('Processing VAL '+p)
    source='/home/ubuntu/fakebusters/data/val/'+p+'/real'
    dest='/home/ubuntu/fakebusters/data/val/'+p
    convert_videos_to_tfrecord(source, dest, 2000, 32, "*.mp4")
    '''
    #---TEST
    print('Processing TEST '+p)
    source='/home/ubuntu/fakebusters/data/test/'+p+'/real'
    dest='/home/ubuntu/fakebusters/data/test/tf_records/'+p
    convert_videos_to_tfrecord(source, dest, 2000, 32, "*.mp4")

    #---TEST
    print('Processing FAKES '+p)
    source='/home/ubuntu/fakebusters/data/fakes/'+p
    dest='/home/ubuntu/fakebusters/data/fakes/tf_records/'+p
    convert_videos_to_tfrecord(source, dest, 2000, 32, "*.mp4")

    #---IMPOSTOR
    print('Processing IMPOSTER '+p)
    source='/home/ubuntu/fakebusters/data/imposter/'+p
    dest='/home/ubuntu/fakebusters/data/imposter/tf_records/'+p
    convert_videos_to_tfrecord(source, dest, 2000, 32, "*.mp4")

    end=time.time()

    print("Duration "+p+" ={}".format(end-start))
    '''




print('Processing:')
start=time.time()
source='/Users/dsrincon/Dropbox/Personal/EDU/Posgrado/masters_usa/MIMS/2019-I/cs282_deep_neural_networks/assignments/project/data/imposter/pelosi'
dest='/Users/dsrincon/Dropbox/Personal/EDU/Posgrado/masters_usa/MIMS/2019-I/cs282_deep_neural_networks/assignments/project/data/imposter/pelosi'
convert_videos_to_tfrecord(source, dest, 100, 32, "*.mp4")
end=time.time()
print("Duration ={}".format(end-start))



import math
import random

def sample_chunks(chunk_list,n_frames_per_video):
    sample_lists=[]
    if len(chunk_list[-1])<n_frames_per_video:
        chunk_list=chunk_list[:-1]

    for chunk in chunk_list:
        chunk_i=random.sample(chunk,n_frames_per_video)
        #step_size=int(math.floor(len(chunk)/n_frames_per_video))
        #chunk_i=[chunk[i] for i in range(0,len(chunk),step_size)]
        chunk_i.sort()
        sample_lists.append(chunk_i)

    return sample_lists

def get_chunks(l, n):
  """Yield successive n-sized chunks from l.
  Used to create n sublists from a list l"""
  for i in range(0, len(l), n):
    yield l[i:i + n]
'''

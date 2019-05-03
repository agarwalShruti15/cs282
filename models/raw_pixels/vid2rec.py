from video2tfrecord_test2 import convert_videos_to_tfrecord
source='/Users/dsrincon/Dropbox/Personal/EDU/Posgrado/masters_usa/MIMS/2019-I/cs282_deep_neural_networks/assignments/project/data/train/michelle/real'
dest='/Users/dsrincon/Dropbox/Personal/EDU/Posgrado/masters_usa/MIMS/2019-I/cs282_deep_neural_networks/assignments/project/data/train/michelle/real'
convert_videos_to_tfrecord(source, dest, 100, 64, "*.mp4")

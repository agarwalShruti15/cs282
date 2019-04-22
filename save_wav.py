import os

#video2flacAudio(input folder, output folder, extension of video files)
def video2wavAudio(in_vid_file, cur_out_file):

    #audio with sampling rate 16KHz and single channel
    if not os.path.exists(cur_out_file):
        cmd = 'ffmpeg -i {0} -af aformat=s16:16000 -ac 1 {1} -hide_banner -loglevel panic'\
            .format(in_vid_file, cur_out_file)
        os.system(cmd)



bsfldr = 'dataset'
for dirpath, dirnames, filenames in os.walk(bsfldr):
    filenames = [f for f in filenames if f.endswith('.mp4')]
    for filename in filenames:
        video2wavAudio(os.path.join(dirpath, filename), os.path.join(dirpath, os.path.splitext(filename)[0] + '.wav'))


import os
import argparse

#extract the features from a single video file
def get_one(i_f, o_f):

    fl_n, ext = os.path.splitext(i_f)
    output = os.path.join(o_f, fl_n + '.csv') #output csv

    try:
        if not os.path.exists(output):
                cmd = './build/bin/FeatureExtraction -f {} -q -out_dir {}'.format(i_f, o_f)
                os.system(cmd)
                os.remove(os.path.join(o_f, fl_n + '_of_details.txt'))

    except Exception as e:
        print(e)

if __name__ == '__main__':

    # Instantiate the parser
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--v', type=str, help='base path to videos, the directory with subject '
                                              'folders inside which are youtube id folders with utterance')
    parser.add_argument('--o', type=str, help='output directory, will keep the same directory structure as input videos')
    parser.add_argument('--vb', type=int, help='verbose, output name of the files', default=1)
    args = parser.parse_args()
    vid_fd = args.v
    out_fd = args.o
    verbose = args.vb

    #create base out dir
    if not os.path.exists(out_fd):
        os.makedirs(out_fd)

    #subject folders
    subject_fldr = [f for f in os.listdir(vid_fd) if os.path.isdir(os.path.join(vid_fd, f))]

    for sf in subject_fldr:

        cur_sf_fldr = os.path.join(vid_fldr, sf)
        vid_fldr = [f for f in os.listdir(cur_sf_fldr) if os.path.isdir(os.path.join(cur_sf_fldr, f))]

        for vid in vid_fldr:

            cur_vid_fldr = os.path.join(cur_sf_fldr, vid)
            cur_out_fldr = os.path.join(out_fd, sf, vid)

            if not os.path.exists(cur_out_fldr):
                os.makedirs(cur_out_fldr)

            v_files = [f for f in os.listdir(cur_vid_fldr) if f.endswith('.mp4')]
            for v in v_files:
                get_one(os.path.join(cur_vid_fldr, v), cur_out_fldr)

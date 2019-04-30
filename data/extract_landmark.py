
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
    parser.add_argument('--i', type=str, help='base path to videos, the directory with subject '
                                              'folders inside which are youtube id folders with utterance')
    parser.add_argument('--o', type=str, help='output directory, will keep the same directory structure as input videos')
    parser.add_argument('--vb', type=int, help='verbose, output name of the files', default=1)
    args = parser.parse_args()
    bs_fldr = args.i
    out_fldr = args.o
    verbose = args.vb

    #create base out dir
    if not os.path.exists(out_fldr):
        os.makedirs(out_fldr)

    #subject folders
    subject_fldr = [f for f in os.listdir(bs_fldr) if os.path.isdir(os.path.join(bs_fldr, f))]

    for sf in subject_fldr:

        if verbose:
            print('\t s: {}'.format(sf))

        cur_sf_fldr = os.path.join(bs_fldr, sf)
        vid_fldr = [f for f in os.listdir(cur_sf_fldr) if os.path.isdir(os.path.join(cur_sf_fldr, f))]

        for vid in vid_fldr:

            if verbose:
                print('\t\t v: {}'.format(vid))

            cur_vid_fldr = os.path.join(cur_sf_fldr, vid)
            cur_out_fldr = os.path.join(out_fldr, sf, vid)

            if not os.path.exists(cur_out_fldr):
                os.makedirs(cur_out_fldr)

            v_files = [f for f in os.listdir(cur_vid_fldr) if f.endswith('.mp4')]
            for v in v_files:

                if verbose:
                    print('\t\t\t u: {}'.format(v))

                get_one(os.path.join(cur_vid_fldr, v), cur_out_fldr)

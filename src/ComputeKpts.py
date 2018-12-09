from KITTIdataset import KITTIdataset
import os, sys
import argparse
from joblib import Parallel, delayed
from multiprocessing import Pool
from pdb import set_trace as st
import collections
import numpy as np
from PIL import Image
from functools import reduce
import pickle
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--data-root-path", type=str, required=True, help="where the dataset is stored")
# parser.add_argument("--dataset-name", type=str, required=True, choices=["kitti_raw_eigen", "kitti_raw_stereo", "kitti_odom", "cityscapes"])
# parser.add_argument("--dump-root", type=str, required=True, help="Where to dump the data")
parser.add_argument("--seq-length", type=int, required=True, help="Length of each training sequence")
parser.add_argument("--img-height", type=int, default=128, help="image height")
parser.add_argument("--img-width", type=int, default=416, help="image width")
parser.add_argument("--min-kpts-num", type=int)
parser.add_argument("--num-threads", type=int, default=4, help="number of threads to use")
args = parser.parse_args()

kpt_num = 300
save_img = True
def computeMatches(image1, image2):
    orb = cv2.ORB_create(nfeatures = 20000)
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    kpts1, des1 = orb.detectAndCompute(gray1, None)
    kpts2, des2 = orb.detectAndCompute(gray2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    matches = sorted(matches, key = lambda x:x.distance)
    max_dist = max(2*matches[0].distance, 30)
    good_matches = [m for m in matches if m.distance < max_dist]

    if save_img:
        image_raw = np.hstack([image1, image2])
        image3 = cv2.drawMatches(image1, kpts1, image2, kpts2, matches, None, flags=2)
        image3 = np.vstack([image_raw, image3])
        cv2.imwrite('images/matches.png', image3)
        good_matches_image = cv2.drawMatches(image1, kpts1, image2, kpts2, good_matches, None, flags=2)
        good_matches_image = np.vstack([image_raw, good_matches_image])
        cv2.imwrite('images/good_matches.png', good_matches_image)
    good_kpts1 = []
    good_kpts2 = []
    for m in good_matches:
        
        p1 = kpts1[m.queryIdx]
        p2 = kpts2[m.trainIdx]
        good_kpts1.append(p1)
        good_kpts2.append(p2)
    good_kpts1 = [kpt.pt for kpt in good_kpts1]
    good_kpts2 = [kpt.pt for kpt in good_kpts2]

    return good_kpts1, good_kpts2

def read_list(list_file, data_dir):
    list_file = os.path.join(data_dir, list_file)

    frame_pathes = []
    with open(list_file) as file:
        for line in file:
            frame_path = line.strip()
            seq_path, frame_name = frame_path.split(" ")
            frame_path = os.path.join(data_dir, seq_path, frame_name+'.jpg')
            frame_pathes.append(frame_path)
    return frame_pathes

def count_kpts(data):
    i, filepath = data
    if i%100==0:
        print("processing %d"%i)
    frames_cat = np.array(Image.open(filepath))
    frame_list = []
    for i in range(args.seq_length):
        frame_list.append(frames_cat[:,i*args.img_width:(i+1)*args.img_width,:])

    src_idx = int((args.seq_length-1)/2)
    kpts_list = []
    kpt_num_list = {}
    all_kpts_pairs = {}
    valid_seq = True
    for i in range(args.seq_length):
        if i!=src_idx:
            kpts1, kpts2 = computeMatches(frame_list[src_idx], frame_list[i])
            l = len(kpts1)
            if l<args.min_kpts_num:
                valid_seq = False
            if l in kpt_num_list.keys():
                kpt_num_list[l] += 1
            else:
                kpt_num_list[l] = 1
            st()
            single_pair = {}
            single_pair[src_idx] = kpts1
            single_pair[i] = kpts2
            all_kpts_pairs[i] = single_pair
    kpts_save_path = filepath.split('.')[0]+'.pickle'
    with open(kpts_save_path, 'wb') as f:
        pickle.dump(all_kpts_pairs, f, protocol=pickle.HIGHEST_PROTOCOL)
    if valid_seq:
        valid_seq = filepath
    else:
        valid_seq = None

    return kpt_num_list, valid_seq

def merge_counts(counts1, counts2):
   for word, count in counts2.items():
      if word not in counts1:
         counts1[word] = 0
      counts1[word] += counts2[word]
   return counts1

def count_one_list(mode):
    assert mode in ['train', 'val']
    img_list = read_list(mode+'.txt', args.data_root_path)

    pool = Pool(processes=args.num_threads)
    # results = pool.map(count_kpts, list(enumerate(img_list)))
    count_kpts((0, img_list[10]))

    per_doc_counts = []
    good_seq = []
    for item in results:
        per_doc_counts.append(item[0])
        if item[1] is not None:
            good_seq.append(item[1])
    counts = reduce(merge_counts, [{}]+per_doc_counts)
    bad_seq = []
    for img_file in img_list:
        if img_file not in good_seq:
            bad_seq.append(img_file)
    with open(os.path.join(args.data_root_path, 'thred=%d'%args.min_kpts_num+'_'+mode+'_kpts_num.txt'), 'w') as f:
        for k,v in sorted(counts.items()):
            f.write( "{}\t\t{}\n".format(k,v) )
    with open(os.path.join(args.data_root_path, 'thred=%d'%args.min_kpts_num+'_'+mode+'_good_seq.txt'), 'w') as f:
        for filename in sorted(good_seq):
            f.write("%s\n"%filename)
    with open(os.path.join(args.data_root_path, 'thred=%d'%args.min_kpts_num+'_'+mode+'_bad_seq.txt'), 'w') as f:
        for filename in sorted(bad_seq):
            f.write("%s\n"%filename)



def main():
    import collections

    count_one_list('train')
    count_one_list('val')





if __name__=='__main__':
    main()

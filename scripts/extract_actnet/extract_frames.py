import os
import subprocess
from tqdm import tqdm
import pickle
import shutil
import argparse
import glob
import json
import os
import cv2
import numpy as np
from glob import glob
import subprocess

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_a', type=str, default='dataset/val_a.json',
                        help='path to the json file containing your prediction')
    parser.add_argument('--train_a', type=str, default='dataset/train_a.json',
                        help='path to the json file containing the ground true')
    parser.add_argument('--test_a', type=str, default='dataset/test_a.json',
                        help='path to the json file containing the ground true')
    parser.add_argument('--val_q', type=str, default='dataset/val_q.json',
                        help='path to the json file containing your prediction')
    parser.add_argument('--train_q', type=str, default='dataset/train_q.json',
                        help='path to the json file containing the ground true')
    parser.add_argument('--test_q', type=str, default='dataset/test_q.json',
                        help='path to the json file containing the ground true')
    parser.add_argument('--video_path', type=str, default='/home/WorkSpace/AVQA/Dataset/ActivityNet/Dataset/ActivityNet/videos',
                    help='path to the json file containing your prediction')
    parser.add_argument('--frame_path', type=str, default='/home/WorkSpace/AVQA/Dataset/ActivityNet/Dataset/ActivityNet/ActNet-frames',
                    help='path to the json file containing your prediction')

    parser.add_argument("--info_path", type=str, default='', help='mapping the video name to the video_id')
    parser.add_argument("--strategy", type=int, default=1, help='0: extract all the frames; 1: need to specify fps and vframes')
    parser.add_argument("--fps", type=str, default='1', help='the number of frames you want to extract within 1 second')
    parser.add_argument("--vframes", type=str, default='60', help='the maximun number of frames you want to extract')

    parser.add_argument("--video_suffix", type=str, default='mp4')
    parser.add_argument("--frame_suffix", type=str, default='jpg')

    params = parser.parse_args()
    return params

def get_filenames_without_extension(directory):
    filenames = []
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            name, _ = os.path.splitext(filename)
            filenames.append(name)
    return filenames

def find_file_by_name_ignoring_extension(directory, target_name):
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            name_without_ext, _ = os.path.splitext(filename)
            if name_without_ext == target_name:
                return filepath 
    return None  

def extract_frames(video, dst, suffix, strategy, fps=5, vframes=60, prefix='', cleanup=False):
    with open(os.devnull, "w") as ffmpeg_log:
        # if os.path.exists(dst) and cleanup:
        #     print(" cleanup: " + dst + "/")
        #     shutil.rmtree(dst)

        os.makedirs(dst, exist_ok=True)
        if strategy == 0:
            video_to_frames_command = ["ffmpeg",
                                       '-y',
                                       '-i', video,  # input file
                                       '-vf', "scale=iw:-1", # input file
                                       '{0}/{1}%05d.{2}'.format(dst, prefix, suffix)]
        else:
            video_to_frames_command = ["ffmpeg",
                                       '-y',
                                       '-i', video,  # input file
                                       '-vf', "scale=iw:-1", # input file
                                       '-r', fps, #fps 5
                                       # '-vframes', vframes,
                                       '{0}/{1}%05d.{2}'.format(dst, prefix, suffix)]
        subprocess.call(video_to_frames_command,
                        stdout=ffmpeg_log, stderr=ffmpeg_log)


if __name__ == '__main__':
    params = parse_opt()

    train_a = json.load(open(params.train_a, 'r'))
    val_a = json.load(open(params.val_a, 'r'))
    test_a = json.load(open(params.test_a, 'r'))
    train_q = json.load(open(params.train_q, 'r'))
    val_q = json.load(open(params.val_q, 'r'))
    test_q = json.load(open(params.test_q, 'r'))

    question = train_q + val_q + test_q
    answer = train_a + val_a + test_a
    videos_file = get_filenames_without_extension(params.video_path)

    if not os.path.exists(params.frame_path):
        os.makedirs(params.frame_path)

    unique_question_ids = set() 
    for qst in question:
        question_id = qst['question_id'].rsplit('_', 1)[0]
        unique_question_ids.add(question_id)  

    question_id_list = list(unique_question_ids)

    count = 0
    video_num = 1
    for question_id in question_id_list:
        print('process video ', video_num)

        if question_id in videos_file:

            dst_frame_path = os.path.join(params.frame_path, question_id)

            if os.path.exists(dst_frame_path):
                continue

            video_path = find_file_by_name_ignoring_extension(params.video_path, question_id)

            extract_frames(
                video=video_path, 
                dst=dst_frame_path, 
                suffix=params.frame_suffix, 
                strategy=params.strategy, 
                fps=params.fps, 
                vframes=params.vframes
            )
            count+=1
        video_num += 1

    print('finished total: ', count)

import os
import sys
import json
import subprocess
import numpy as np
import torch
from torch import nn

from opts import parse_opts
from model import generate_model
from mean import get_mean
from classify import classify_video
from tqdm import tqdm
from pathlib import Path
import shutil

if __name__ == "__main__":
    opt = parse_opts()
    opt.mean = get_mean()
    opt.arch = '{}-{}'.format(opt.model_name, opt.model_depth)
    opt.sample_size = 112
    opt.sample_duration = 64
    opt.n_classes = 400

    model = generate_model(opt)
    print('loading model {}'.format(opt.model))
    model_data = torch.load(opt.model)
    assert opt.arch == model_data['arch']
    model.load_state_dict(model_data['state_dict'])
    model.eval()
    if opt.verbose:
        print(model)

    input_files = []
    with open(opt.input, 'r') as f:
        for row in f:
            input_files.append(row[:-1])

    class_names = []
    with open('class_names_list') as f:
        for row in f:
            class_names.append(row[:-1])

    ffmpeg_loglevel = 'quiet'
    if opt.verbose:
        ffmpeg_loglevel = 'info'

    tmp_path = Path('tmp')
    if tmp_path.exists():
        shutil.rmtree(tmp_path)

    outputs = []
    for input_file in input_files:
        video_path = os.path.join(opt.video_root, input_file)
        if os.path.exists(video_path):
            print(video_path)
            tmp_path.mkdir()
            subprocess.call(
                'ffmpeg -i {} tmp/image_%06d.jpg'.format(video_path),
                shell=True)

            result = classify_video('tmp', input_file, class_names, model, opt)
            outputs.append(result)

            shutil.rmtree(tmp_path)
        else:
            print('{} does not exist'.format(input_file))

    with open(opt.output, 'w') as f:
        json.dump(outputs, f)

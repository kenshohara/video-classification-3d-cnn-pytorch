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
from spatial_transforms import (Compose, Normalize, Scale, CenterCrop, ToTensor)
from temporal_transforms import LoopPadding, TemporalRandomCrop

if __name__=="__main__":
    opt = parse_opts()
    opt.mean = get_mean(opt.norm_value)
    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    opt.sample_size = 112
    opt.sample_duration = 16
    opt.n_classes = 400

    model = generate_model(opt)
    print('loading checkpoint {}'.format(opt.resume_path))
    model_data = torch.load(opt.resume_path)
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

    if os.path.exists('tmp'):
        subprocess.call('rm -rf tmp', shell=True)

    outputs = []
    for input_file in input_files:
        if os.path.exists(input_file):
            print(input_file)
            subprocess.call('mkdir tmp', shell=True)
            subprocess.call('ffmpeg -i {} tmp/image_%05d.jpg'.format(input_file), shell=True)

            result = classify_video('tmp', input_file, class_names)
            outputs.append(result)

            subprocess.call('rm -rf tmp', shell=True)
        else:
            print('%s does not exist'.format(input_file))

    if os.path.exists('tmp'):
        subprocess.call('rm -rf tmp', shell=True)

    with open(opt.output, 'w') as f:
        json.dump(outputs, f)

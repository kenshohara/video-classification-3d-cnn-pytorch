# Video Classification Using 3D ResNet
This is a pytorch code for video (action) classification using 3D ResNet trained by [this code](https://github.com/kenshohara/3D-ResNets-PyTorch).  
The 3D ResNet is trained on the Kinetics dataset, which includes 400 action classes.  
This code uses videos as inputs and outputs class names and predicted class scores for each 16 frames in the score mode.  
In the feature mode, this code outputs features of 512 dims (after global average pooling) for each 16 frames.  

**Torch (Lua) version of this code is available [here](https://github.com/kenshohara/video-classification-3d-cnn).**

## Requirements
* [PyTorch](http://pytorch.org/)
```
conda install pytorch torchvision cuda80 -c soumith
```
* FFmpeg, FFprobe
```
wget http://johnvansickle.com/ffmpeg/releases/ffmpeg-release-64bit-static.tar.xz
tar xvf ffmpeg-release-64bit-static.tar.xz
cd ./ffmpeg-3.3.3-64bit-static/; sudo cp ffmpeg ffprobe /usr/local/bin;
```
* Python 3

## Preparation
* Download this code.
* Download the [pretrained model](https://github.com/kenshohara/3D-ResNets-PyTorch/releases).  

## Usage
Assume input video files are located in ```./videos```.

To calculate class scores for each 16 frames, use ```--mode score```.
```
python main.py --input ./input --video_root ./videos --output ./output.json --model ./resnet-34-kinetics.pth --mode score
```
To visualize the classification results, use ```generate_result_video/generate_result_video.py```.

To calculate video features for each 16 frames, use ```--mode feature```.
```
python main.py --input ./input --video_root ./videos --output ./output.json --model ./resnet-34-kinetics.pth --mode feature
```


## Citation
If you use this code, please cite the following:
```
@article{hara3dresnets
  author={Kensho Hara and Hirokatsu Kataoka and Yutaka Satoh}
  title={Learning Spatio-Temporal Features with 3D Residual Networks for Action Recognition}
  journal={arXiv preprint}
  volume={arXiv:1708.07632}
  year={2017}
}
```

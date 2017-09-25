# Result Video Generation
This is a code for generating videos of classification results.  
It uses both ```output.json``` and videos as inputs and draw predicted class names in each frame.

## Requirements
* Python 3
* Pillow
* ffmpeg, ffprobe

## Usage
To generate videos based on ```../output.json```, execute the following.
```
python generate_result_video.py ../output.json ../videos ./videos_pred ../class_names_list 5
```
The 2nd parameter (```../videos```) is the root directory of videos.
The 3rd parameter (```./videos_pred```) is the directory path of output videos.
The 5th parameter is a size of temporal unit.  
The CNN predicts class scores for a 16 frame clip.  
The code averages the scores over each unit.  
The size 5 means that it averages the scores over 5 clips (i.e. 16x5 frames).  
If you use the size as 0, the scores are averaged over all clips of a video.  

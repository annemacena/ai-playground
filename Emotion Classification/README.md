# Emotion Classification

Simple python app for emotion classification. A pipeline was followed where the image or video passes through a pre-trained YOLOv5 detector to detect faces and then to a pre-trained ResNet50 classifier. 

- The detector was pre-trained on imagenet and then the last layers were trained on [WIDER dataset](http://shuoyang1213.me/WIDERFACE/). Training process can be found [here](https://colab.research.google.com/drive/1YUDGr3vVX7dGF92WI96feImw_BD4EVsK?usp=sharing).
- The classifier was pre-trained on imagenet and then the last layers were trained on [fer2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013). Training process can be found [here](https://colab.research.google.com/drive/1ezTvWQuDIhmSFGu6ZuZ02AYxKZnK5IJ-?usp=sharing).

<div align="center">
  <p>⠀</p>
  <img src="sample.gif" />
  <p>⠀</p>
</div>

# Requirements

You can found all necessary libs in `requirements.txt`.

- Create a conda env (put your env name where is "env")
```bash
conda create --name <env> --file requirements.txt
```

- Activate conda env
```bash
conda activate <env>
```

⚠️ The model weights obtained on training processes should be downloaded and placed in root of the project or you can set the paths on line 21 of `pipeline.py`.

- [Detector weights](https://drive.google.com/file/d/1RL1HzCmL6oq3t0GMUyLPxEG_nDafi5PM/view?usp=sharing).
- [Classifier weights](https://drive.google.com/file/d/1-k-knsG7hFEZ4x3RlUiqDPUJAb3hqP9c/view?usp=sharing).

⚠️ The classifier classes (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral) can be found in `classes.csv`. It is important to know that the classifier has been trained on these classes specified in the file, so if you want to add more classes, you should either do the training process with your new classes or implement some metric learning technique.

## Command example

```bash
python main.py --input samples/image_test.jpg
```

There are others arguments beside `input`:
- `save`: saves the output video/image on `results/` directory
- `debug`: shows the output image or shows output video frame by frame

# To-do

- [ ] Optimize classifier accuracy
  - [ ] Train classifier on AffectNet dataset
  - [ ] Test pipeline using only the detector
- [ ] Implement webapp with the pipeline proposed

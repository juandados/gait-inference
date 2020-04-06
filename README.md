# Intro
## Citation
```
@article{juandados2020gait-inference,
  title={Gait Inference},
  author={Chacon, Juan},
  journal={GitHub Repository. Available online: https://github.com/juandados/gait-inference (accessed on dd Month Year)},
  year={2020}
}
```
# Getting Started
## Setting Up Environment
This works with python 3.6. Install the specified requierements (e.g. using ```pip install -r requirements.txt```)
## Setting Up Darkflow
We use [darkflow](https://github.com/thtrieu/darkflow.git) a python implementation of YOLO for human detection. **Note:** All the steps in this section are relative to the darkflow repo directory.
1. Clone the darkflow [repository](https://github.com/thtrieu/darkflow.git) inside the gait-inference repo.
2. Download the correct weigths for yolov2 (check weights section below). Save the .cfg file in the cfg folder and the .weights file in the bin folder (this folder should be created).
```
mkdir bin
curl -o cfg/yolov2-tiny.cfg https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov2-tiny.cfg
curl -o bin/yolov2-tiny.weights https://pjreddie.com/media/files/yolov2-tiny.weights
```
3. Include the model name (e.g. yolov2-tiny) in the coco_models list in the file `darkflow/net/yolo/misc.py`
4. (Only for tiny) change self.offset=20 (for yolov2-tiny) in the file `darkflow/utils/loader.py`
5. Install darkflow using ```pip install .```

**Weights**: The correct weigths and cfg files pairs are
* https://pjreddie.com/media/files/yolov2.weights, https://github.com/pjreddie/darknet/blob/master/cfg/yolov2.cfg
* https://pjreddie.com/media/files/yolov2-tiny.weights, https://github.com/pjreddie/darknet/blob/master/cfg/yolov2-tiny.cfg

## Training Notebook
The notebook [code/gait_inference.ipynb](https://github.com/juandados/gait-inference/blob/master/code/gait_inference.ipynb) contains some basic steps for training the pose estimator and gait prediction models.

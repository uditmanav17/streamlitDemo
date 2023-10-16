# NFL Players Collision Detections


## Introduction | Problem Statement
The objective is to detect the number of helmet collisions players have had on field during a session from the session's video, partly inspired by [NFL Health & Safety: Helmet Assignment](https://www.kaggle.com/c/nfl-health-and-safety-helmet-assignment).


## Dataset Description
We'll be using a subset of the dataset provided [here](https://www.kaggle.com/competitions/nfl-health-and-safety-helmet-assignment/data). Specifically, we'll be using the `images` directory and `image_labels.csv` to train a hemlet detection model.
- `images` directory contains `9947 images` from the field from different angles.
- `image_labels.csv` contains `~193K rows` describing bounding boxes corresponding to images.


## Approach and Model Selection
Since we are dealing with object detection problems, we can try various detection models like [FastRCNN](https://arxiv.org/abs/1504.08083), [FasterRCNN](https://arxiv.org/abs/1506.01497), [RetinaNet](https://paperswithcode.com/method/retinanet), [YOLOv5](https://pytorch.org/hub/ultralytics_yolov5/), [YOLOv8](https://docs.ultralytics.com/models/yolov8/) or [other options](https://huggingface.co/models?pipeline_tag=object-detection).

For this particular use case, let's go ahead with YOLOv8 as it supports different modes, such as object tracking, which will be useful when accurately detecting the number of helmet collisions.


## Pre-processing Dataset
As with any model training pipeline, we will first have to perform EDA, then prepare/process data in a format compatible with model input. The following steps were performed as a part of EDA and data pre-processing:
- Download dataset from kaggle
- Check a sample image with bounding boxes
- Split into train, test and validation set based on images
- Scale bounding box annotations as per YOLOv8 input as described [here](https://docs.ultralytics.com/datasets/detect/)

Please refer to [eda.ipynb](https://github.com/uditmanav17/streamlitDemo/blob/nfl/nfl-player-collision-detection/eda.ipynb) for detailed code.


## Model Training
Once the dataset is transformed into model input compatible format, we create a model configuration YAML file, describing `train`, `test`, `valid` directory paths, and `num_classes` with `names` (class mappings). Refer to [this](https://docs.ultralytics.com/datasets/detect/) for a sample configuration file.

Once model configuration is defined, we can start model training via the Python API
```
results = model.train(data='config.yaml', epochs=100)
```
or CLI
```
yolo train model=yolov8n.pt data=config.yaml epochs=100
```
More details on training model can be found [here](https://docs.ultralytics.com/models/yolov8/#usage).


## Deploy
Once satisfied with the model's accuracy, it was wrapped up in a basic [streamlit](https://streamlit.io/) application and deployed on [render](https://render.com/). Deployment code can be found [here](https://github.com/uditmanav17/streamlitDemo/blob/nfl/nfl-player-collision-detection/streamlit_app.py).

The deployed application endpoint can be found [here](https://nfl-collision-count.onrender.com/). Please note that the application endpoint may take a few minutes to start.


## Application Demo
<p align="center">
<iframe allowfullscreen="allowfullscreen" src="https://drive.google.com/file/d/1Fx74MfKMp68kK7JljwYiFaJD_QTwgOP1/preview" width="960" height="540" allow="autoplay"></iframe>
</p>
<!-- ## Possible Future Improvements -->



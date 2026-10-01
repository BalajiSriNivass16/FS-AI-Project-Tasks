
ROBOFLOW DATASET LINK : https://universe.roboflow.com/-jwzpw/continuous_fire/dataset/5

[USING YOLOV8 ENGINE]

ACESSING DATASET WITH API IN ROBOFLOW:


["!pip install roboflow
from roboflow import Roboflow
rf = Roboflow(api_key="MlzksBF9LmPSy0mRrwCI")
project = rf.workspace("-jwzpw").project("continuous_fire")
version = project.version(5)
dataset = version.download("yolov8")"]
                
TRAIN DATASET AND DOWLOAD A BEST.pt WITH COLAB 
after train a dataset use the data model

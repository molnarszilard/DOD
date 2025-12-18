from ultralytics import YOLO
# import os
# import inspect
# print(inspect.getfile(YOLO))

# Load a model
# model = YOLO('yolov8m-obb.yaml')  # build a new model from YAML
model = YOLO('yolov8m-seg.pt')  # load a pretrained model (recommended for training)
# model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

# Train the model
results = model.train(
    cfg='ultralytics/cfg/default.yaml',
    data='ultralytics/cfg/datasets/leaf_segment.yaml',
    mode='train',
    task='segment',
    batch=4,
    epochs=100, 
    imgsz=1024,
)
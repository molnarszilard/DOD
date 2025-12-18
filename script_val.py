from ultralytics import YOLO


# DIRTYPE = "angle" # Direction type is an angle (1 param)
DIRTYPE = "psc" # Direction type is an angle (3param encoding aka phase-shifting coder)

# Load a model
model = YOLO('runs/seg/train/weights/best.pt')

# Validate the model
metrics = model.val(
    cfg='ultralytics/cfg/default.yaml',
    # data='ultralytics/cfg/datasets/vedai.yaml',
    data='ultralytics/cfg/datasets/OHD-SJTU-L.yaml',
    task='dod',
    batch=4,
    imgsz=1024,
    dir_type=DIRTYPE,
)

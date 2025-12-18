from ultralytics import YOLO

# DIRTYPE = "angle" # Direction type is an angle (1 param)
DIRTYPE = "psc" # Direction type is an angle (3param encoding aka phase-shifting coder)

# loss of direction point (Options: mahalanobis,euclidean,vector,probiou,angle,kl,gma)
# DIRLOSS = "vector"
DIRLOSS = "probiou"
# DIRLOSS = "angle"

# Load a model
model = YOLO('runs/dod/train/weights/best.pt')

# Run batched inference on a list of images
results = model(
    source='path_to_images',
    task='dod',
    imgsz=1024,
    save_txt=True,
    iou=0.4,
    dir_type=DIRTYPE
)

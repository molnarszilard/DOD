from ultralytics import YOLO

# DIRTYPE = "angle" # Direction type is an angle (1 param)
DIRTYPE = "psc" # Direction type is an angle (3param encoding aka phase-shifting coder)

# loss of direction point (Options: mahalanobis,euclidean,scalarprod,probiou,angle,kl,gma)
# DIRLOSS = "scalarprod"
DIRLOSS = "probiou"
# DIRLOSS = "angle"

if DIRTYPE == 'angle':
    model = YOLO('yolov8m-dod_angle.yaml')
elif DIRTYPE == 'psc':
    model = YOLO('yolov8m-dod_psc.yaml')
else:
    model = YOLO('yolov8m-dod.yaml')

# Train the model
results = model.train(
    cfg='ultralytics/cfg/default.yaml',
    data='ultralytics/cfg/datasets/leaf_dod.yaml',
    # data='ultralytics/cfg/datasets/OHD-SJTU-L.yaml',
    mode='train',
    task='dod',
    epochs=100, 
    imgsz=1024,
    batch=4,
    direction_gain=5,
    cepoch=-1, # -1 no dynamic direction_ loss, 0 dinamic direction_ loss
    direction_loss=DIRLOSS,
    dir_type=DIRTYPE,
    optimizer='SGD',
)
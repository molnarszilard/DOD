from ultralytics import YOLO

# Load a model
model = YOLO('runs/seg/train/weights/best.pt')

# Run batched inference on a list of images
results = model(
    source='path_to_images',
    imgsz=1024,
    save_txt=True,
    iou=0.4,
    show_labels=False
)
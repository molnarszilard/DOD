### This script reads the annotation files of the DOTA dataset (in the original format: cls x1 y1 x2 y2 x3 y3 x4 y4 class_name difficulty -- coordinates are in pixel)
### And converts it back to the YOLO format (cls x1 y1 x2 y2 x3 y3 x4 y4 -- coordinates are normalized between [0,1])
import os
import argparse
import cv2

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--images', default="",
                    help='the directory to the images')
parser.add_argument('--labels', default="",
                    help='the directory to the labels')
parser.add_argument('--labels_new', default="",
                    help='the directory to the labels')
args = parser.parse_args()

classes = ['plane', 'ship', 'large-vehicle', 'small-vehicle', 'helicopter','harbor']

labels=[]
directory=args.labels
dlist=os.listdir(directory)
dlist.sort()
for filename in dlist:
    if filename.endswith(".txt") and not filename.startswith("classes"):
        labels.append(filename)
    else:
        continue

print("Number of labels: "+str(len(labels)))

if args.labels_new is None:
    args.labels_new = os.path.join(args.labels, "new")
if not os.path.exists(args.labels_new):
    os.makedirs(args.labels_new)

for i in range(len(labels)):
    label_name=labels[i]
    img_extensions=["png","jpg","tif"]
    img = None
    for img_ext in img_extensions:        
        if os.path.isfile(os.path.join(args.images, label_name[:-3]+img_ext)):
            image_name=label_name[:-3]+img_ext
            img = 0
    if img is None:
        print("Image %spng/jpg/tif does not exists"%(label_name[:-3]))
        continue
    print("Processing: "+label_name)
    savename_label = os.path.join(args.labels_new, label_name)

    f_label = open(os.path.join(args.labels, label_name), "r")
    Lines = f_label.readlines()
    f_label.close()

    f = open(savename_label, "w")
    if len(Lines)<1:
        print("No label.")
        f.close()
        continue
    img = cv2.imread(os.path.join(args.images, image_name))
    H,W,_=img.shape
    for line in Lines: 
        if line.startswith('#') or line.startswith('YOLO'):
            continue
        # class,x_center,y_center,bb_x,bb_y,angle
        # class_index, x1, y1, x2, y2, x3, y3, x4, y4
        current_line = line[:-1]
        elements=current_line.split(" ")
        cls = classes.index(elements[10])
        x1 = float(elements[0])/W
        y1 = float(elements[1])/H
        x2 = float(elements[2])/W
        y2 = float(elements[3])/H
        x3 = float(elements[4])/W
        y3 = float(elements[5])/H
        x4 = float(elements[6])/W
        y4 = float(elements[7])/H
        f.write('%d %f %f %f %f %f %f %f %f\n'%(cls,x1,y1,x2,y2,x3,y3,x4,y4))
    f.close()
        
    
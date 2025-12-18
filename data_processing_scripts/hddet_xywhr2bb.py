import os
import argparse
import cv2
import numpy as np
import math
import utils

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--labels',required=True, default=None,
                    help='the directory to the labels')
parser.add_argument('--images', default=None,
                    help='the directory to the images')
parser.add_argument('--labels_out',required=True, default=None,
                    help='the directory to the labels to save')
parser.add_argument('--pixel', default=False, action='store_true',
                    help='are to coordinates in pixel values?')
parser.add_argument('--change_class', default=False, action='store_true',
                    help='are the class numbers corectly matched?')
args = parser.parse_args()

labels=[]
dlist=os.listdir(args.labels)
dlist.sort()
for filename in dlist:
    if filename.endswith(".txt") and not filename.startswith("classes"):
        #print(os.path.join(directory, filename))
        labels.append(filename)
    else:
        continue
if len(labels)<1:
    print("%s is empty"%(args.labels))
    exit()

new_label_dir=args.labels_out
if not os.path.exists(new_label_dir):
    os.makedirs(new_label_dir)

class_change = ['5','4','2','0','1','3']
#### yolo classes names:
  # 0: plane
  # 1: ship
  # 2: vechL
  # 3: vechS
  # 4: heli
  # 5: harbor
  # what is X in HDDet change to Y in YOLO
  # 0-5, 1-4, 2-2, 3-0, 4-1, 5-3

if __name__ == '__main__':
    occluded = 0
    visible = 0
    for i in range(len(labels)):
        label_name=labels[i]
        print("Processing: "+label_name)
        f_label = open(os.path.join(args.labels, label_name), "r")
        Lines = f_label.readlines()
        f_label.close()
        
        img_extensions=["png","jpg","tif"]
        img = None
        for img_ext in img_extensions:
            if os.path.isfile(os.path.join(args.images, label_name[:-3]+img_ext)):
                img_name=label_name[:-3]+img_ext
                # print(img_name)
                img = cv2.imread(os.path.join(args.images, img_name))
        if img is None:
            print("Image %s does not exists"%(label_name[:-3]))
            continue
        H,W,_=img.shape
        fl = open(os.path.join(args.labels_out, label_name), "w")
        for line in Lines:
            if line.startswith('#') or line.startswith('YOLO'):
                continue
            # class,x_center,y_center,bb_x,bb_y,angle
            current_line = line[:-1]
            elements=current_line.split(" ")

            if args.pixel:
                cx=float(elements[1])
                cy=float(elements[2])
                w=float(elements[3])
                h=float(elements[4])
            else:
                cx=float(elements[1])*W
                cy=float(elements[2])*H
                w=float(elements[3])*W
                h=float(elements[4])*H
            cl = int(elements[0])
            if args.change_class:
                cl=class_change[cl]
            angle=math.radians(float(elements[5]))
            a=min(w,h)/2
            dx=cx+a*math.cos(angle-math.pi/2)
            dy=cy+a*math.sin(angle-math.pi/2)
            corners = utils.xywhr2xyxyxyxy(np.asarray([cx,cy,w,h,angle]))
            if corners[0,0]<0:
                corners[0,0]=0
            if corners[0,1]<0:
                corners[0,1]=0
            if corners[1,0]<0:
                corners[1,0]=0
            if corners[1,1]<0:
                corners[1,1]=0
            if corners[2,0]<0:
                corners[2,0]=0
            if corners[2,1]<0:
                corners[2,1]=0
            if corners[3,0]<0:
                corners[3,0]=0
            if corners[3,1]<0:
                corners[3,1]=0
            if corners[0,0]>=W:
                corners[0,0]=W-1
            if corners[0,1]>=H:
                corners[0,1]=H-1
            if corners[1,0]>=W:
                corners[1,0]=W-1
            if corners[1,1]>=H:
                corners[1,1]=H-1
            if corners[2,0]>=W:
                corners[2,0]=W-1
            if corners[2,1]>=H:
                corners[2,1]=H-1
            if corners[3,0]>=W:
                corners[3,0]=W-1
            if corners[3,1]>=H:
                corners[3,1]=W-1
            
            # fl.write("%s %f %f %f %f %f %f %f %f %f %f\n"%(cl,corners[0,0]/W,corners[0,1]/H,corners[1,0]/W,corners[1,1]/H,corners[2,0]/W,corners[2,1]/H,corners[3,0]/W,corners[3,1]/H,dx/W,dy/H))
            fl.write("%s %f %f %f %f %f %f %f %f %f %f\n"%(cl,corners[0,0]/W,corners[0,1]/H,corners[1,0]/W,corners[1,1]/H,corners[2,0]/W,corners[2,1]/H,corners[3,0]/W,corners[3,1]/H,math.cos(angle),math.sin(angle)))
        fl.close()
            
            
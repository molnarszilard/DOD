import os
import numpy as np
import argparse
import math
import cv2
import utils

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--images', default=None,
                    help='the directory to the images')
parser.add_argument('--labels', default=None,
                    help='the directory to the labels')
parser.add_argument('--new_labels', default=None,
                    help='the directory where to save the new labels')
parser.add_argument('--new_images', default=None,
                    help='where to save the new images')
parser.add_argument('--test', default=False, action='store_true',
                    help='save the plotted images?')
parser.add_argument('--mode', default='ud',
                    help='flip upside-down: ud, or left-right: lr')
args = parser.parse_args()


def parsePoints(elements, H, W, image=None):
    Ax = float(elements[1])
    Ay = float(elements[2])
    Bx = float(elements[3])
    By = float(elements[4])
    Cx = float(elements[5])
    Cy = float(elements[6])
    Dx = float(elements[7])
    Dy = float(elements[8])
    Dpx = float(elements[9])
    Dpy = float(elements[10])

    if args.mode in ['lr']:
        Ax = 1-Ax
        Bx = 1-Bx
        Cx = 1-Cx
        Dx = 1-Dx
        Dpx = 1-Dpx
    elif args.mode in ['ud']:
        Ay = 1-Ay
        By = 1-By
        Cy = 1-Cy
        Dy = 1-Dy
        Dpy = 1-Dpy
    if image is not None:
        image = cv2.line(image, (int(Ax*W),int(Ay*H)), (int(Bx*W),int(By*H)),utils.get_color('pink'), 2)
        image = cv2.line(image, (int(Bx*W),int(By*H)), (int(Cx*W),int(Cy*H)),utils.get_color('cyan'), 2)
        image = cv2.line(image, (int(Cx*W),int(Cy*H)), (int(Dx*W),int(Dy*H)),utils.get_color('cyan'), 2)
        image = cv2.line(image, (int(Dx*W),int(Dy*H)), (int(Ax*W),int(Ay*H)),utils.get_color('cyan'), 2)
        image = cv2.circle(image, (int(Dpx*W),int(Dpy*H)), 1, utils.get_color('black'), 3)
        image = cv2.circle(image, (int(Dpx*W),int(Dpy*H)), 1, utils.get_color('yellow'), 2)
    return '%s %f %f %f %f %f %f %f %f %f %f\n'%(elements[0],Ax,Ay,Bx,By,Cx,Cy,Dx,Dy,Dpx,Dpy)
            
images=[]
labels=[]
dlist=os.listdir(args.images)
dlist.sort()
for filename in dlist:
    if filename.endswith(".png") or filename.endswith(".jpg"):
        #print(os.path.join(directory, filename))
        images.append(filename)
    else:
        continue
if len(images)<1:
    print("%s is empty"%(args.images))
    exit()

if not os.path.exists(args.new_images):
        os.makedirs(args.new_images)

if args.labels is not None:
    dlist=os.listdir(args.labels)
    dlist.sort()
    for filename in dlist:
        if filename.endswith(".txt") and filename!="classes.txt":
            #print(os.path.join(directory, filename))
            labels.append(filename)
        else:
            continue
    print("Number of images: "+str(len(images)))
    print("Number of labels: "+str(len(labels)))
    if len(images)!=len(labels) and not args.noneq:
        print("The size of the images and labels folders do not match.")
        exit()

    if not os.path.exists(args.new_labels):
        os.makedirs(args.new_labels)

for i in range(len(images)):
    image_name=images[i]
    print("Processing: "+image_name)
    img = cv2.imread(os.path.join(args.images, image_name))
    if args.mode in ['lr']:
        img_flipped = cv2.flip(img,1)
    elif args.mode in ['ud']:
        img_flipped = cv2.flip(img,0)
    file_name, file_extension = os.path.splitext(image_name)
    savename_flip=os.path.join(args.new_images,file_name+'_'+args.mode+file_extension)
    savename_label = os.path.join(args.new_labels, file_name+'_'+args.mode+".txt")
    f_label = open(os.path.join(args.labels, file_name+".txt"), "r")
    Lines = f_label.readlines()
    f_label.close()
    f = open(savename_label, "w")
    if len(Lines)<1:
        print("No label.")
        f.close()
        continue
    H,W,_ = img.shape

    # f.write("YOLO_OBB\n")
    for line in Lines: 
        # class,x_center,y_center,bb_x,bb_y,angle
        # class_index, x1, y1, x2, y2, x3, y3, x4, y4
        current_line = line[:-1]
        elements=current_line.split(" ")
        noe = len(elements)
        
        if args.test:
            new_line = parsePoints(elements,H,W,image=img_flipped)
        else:
            new_line = parsePoints(elements,H,W)
        if new_line is None:
            continue
        else:
            f.write(new_line)

    f.close()
    cv2.imwrite(savename_flip,img_flipped)

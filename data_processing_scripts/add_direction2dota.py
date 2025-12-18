"""
This script adds a point for each object in the DOTA dataset.
This point is the middle point of between the first and second points, which supposedly represents the direction of the object (not always the case).
Possible inputs: labels, labels_out, images, plots
For a check, the bounding boxes and direction points are plotted on the image.
THe script uses multithreads to speed up the process.
"""


import os
import argparse
import cv2
import math
import utils

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--labels', default="",
                    help='the directory to the labels')
parser.add_argument('--labels_out', default="",
                    help='the directory where to save the labels')
parser.add_argument('--images', default="",
                    help='the directory to the images')
parser.add_argument('--plots', default="",
                    help='the directory where to save plots')

def process(labels,args):
    for i in range(len(labels)):
        label_name=labels[i]
        print("Processing: "+label_name)
        f_label = open(os.path.join(args.labels, label_name), "r")
        Lines = f_label.readlines()
        f_label.close()
        savename_label = os.path.join(args.labels_out, label_name)
        if os.path.exists(savename_label):
            print("File already exists")
            print(label_name)
        f = open(savename_label, "w")
        img_extensions=["png","jpg","tif"]
        img = None
        for img_ext in img_extensions:
            if os.path.isfile(os.path.join(args.images, label_name[:-3]+img_ext)):
                img_name=label_name[:-3]+img_ext
                img = cv2.imread(os.path.join(args.images, img_name))
        if img is None:
            print("Image %s does not exists"%(label_name[:-3]))
            continue
        H,W,_ = img.shape
        for line in Lines:
            if line.startswith('#') or line.startswith('YOLO'):
                continue
            current_line = line[:-1]
            elements=current_line.split(" ")
            if elements is None:
                print("Label %d in %s is wrong, remove it"%(i,label_name))
                continue
            if elements==-1:
                continue
            if len(elements)!=9:
                print("label points wrong")
                print(label_name)
            for i_el in range(len(elements)):
                f.write("%s "%(elements[i_el]))
            Ax = float(elements[1])*W
            Ay = float(elements[2])*H
            Dx = float(elements[7])*W
            Dy = float(elements[8])*H
            direction = math.atan2(Ay-Dy,Ax-Dx)
            f.write("%f %f\n"%(math.cos(direction),math.sin(direction)))

            img = utils.draw_rectangle(img,[elements],colors=['cls','cls','cls','cls','yolored'],thickness=2,plot_kp="kp",norm=True)
        if args.plots is not None:
            cv2.imwrite(os.path.join(args.plots,img_name),img)
        f.close

if __name__ == '__main__':
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

    if args.labels_out is None:
        args.labels_out = os.path.join(args.labels, "labels_combined")
    if not os.path.exists(args.labels_out):
        os.makedirs(args.labels_out)

    if args.plots is not None and not os.path.exists(args.plots):
        os.makedirs(args.plots)
    process(labels,args)
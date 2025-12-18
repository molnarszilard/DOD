import os
import argparse
import cv2
import numpy as np
import math
import utils

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--input', default=None,
                    help='the directory of the input labels')
parser.add_argument('--output', default=None,
                    help='the directory of the output labels')
parser.add_argument('--images', default=None,
                    help='the directory to the images')
parser.add_argument('--plots', default=None,
                    help='the directory where to save plots')
args = parser.parse_args()

labels=[]
dlist=os.listdir(args.input)
dlist.sort()
for filename in dlist:
    if filename.endswith(".txt") and not filename.startswith("classes"):
        #print(os.path.join(directory, filename))
        labels.append(filename)
    else:
        continue
if len(labels)<1:
    print("%s is empty"%(args.input))
    exit()

if args.plots is not None and not os.path.exists(args.plots):
    os.makedirs(args.plots)

if args.output is not None and not os.path.exists(args.output):
    os.makedirs(args.output)

# cl_smallveh = [3,5,7,9] # car,pickup, tractor, vans
# cl_largeveh = [2,8] # campingcar, truck
# cl_ship = [1] # boat
# cl_plane = [6] # plane

cl_smallveh = [1,4,9,11] # car-1, 4-tractor, 9-van, 11-pickup
cl_largeveh = [2,5] # truck-2, campingcar-5
cl_ship = [23] # ship-23
cl_plane = [31] # plane-31
# cl_other = [10] # plane-31
## 10-vehicle? - other?

if __name__ == '__main__':
    for i in range(len(labels)):
        label_name=labels[i]
        print("Processing: "+label_name)
        f_label = open(os.path.join(args.input, label_name), "r")
        Lines = f_label.readlines()
        f_label.close()
        f = open(os.path.join(args.output, label_name[:-4]+"_co.txt"), "w")
        img = None
        # print(args.images)
        if args.images is not None:
            img_extensions=["png","jpg","tif"]
            img = None
            for img_ext in img_extensions:
                # print(os.path.join(args.images, label_name[:-4]+"_co."+img_ext))
                if os.path.isfile(os.path.join(args.images, label_name[:-4]+"_co."+img_ext)):
                    img_name=label_name[:-4]+"_co."+img_ext
                    # print(img_name)
                    img = cv2.imread(os.path.join(args.images, img_name))
            if img is None:
                print("Image %s does not exists"%(label_name[:-4]+"_co."))
                continue
        if img is None:
            f.close()
            continue
        
        for line in Lines:
            if line.startswith('#') or line.startswith('YOLO'):
                continue
            # class,x_center,y_center,bb_x,bb_y,angle
            current_line = line[:-1]
            orig_elements=current_line.split(" ")
            if len(orig_elements)!=14:
                continue
            for i_el in range(len(orig_elements)):
                orig_elements[i_el] = float(orig_elements[i_el])
            H,W,_ = img.shape
            elements = []
            cx = orig_elements[0]
            cy = orig_elements[1]
            angle = orig_elements[2]
            category = int(orig_elements[3])
            occluded = int(orig_elements[4])
            truncated = int(orig_elements[5])
            # if occluded or truncated:
            #     continue

            #### Save in yolo format
            # if category in cl_smallveh:
            #     continue
            # elif category in cl_largeveh:
            #     continue
            # elif category in cl_ship:
            #     continue
            # elif category in cl_plane:
            #     continue
            # else:
            #     elements.append(0)

            #### Save in yolo format
            if category in cl_smallveh:
                elements.append(3)
            elif category in cl_largeveh:
                elements.append(2)
            elif category in cl_ship:
                elements.append(1)
            elif category in cl_plane:
                elements.append(0)
                angle+=math.pi/2
            else:
                continue
            
            elements.append(orig_elements[6]/W)
            elements.append(orig_elements[10]/H)
            elements.append(orig_elements[7]/W)
            elements.append(orig_elements[11]/H)
            elements.append(orig_elements[8]/W)
            elements.append(orig_elements[12]/H)
            elements.append(orig_elements[9]/W)
            elements.append(orig_elements[13]/H)
            elements.append(math.cos(angle))
            elements.append(math.sin(angle))
            elements_plot = elements.copy()
            rect = np.array([int(orig_elements[6]),int(orig_elements[10]),int(orig_elements[7]),int(orig_elements[11]),int(orig_elements[8]),int(orig_elements[12]),int(orig_elements[9]),int(orig_elements[13])])
            # print(elements[1:9])
            # print(rect)
            rect_xywhr = cv2.minAreaRect(rect.reshape(4, 2))
            px,py=utils.dptFromAngle(rect_xywhr[0][0],rect_xywhr[0][1],rect_xywhr[1][0],rect_xywhr[1][1],math.radians(rect_xywhr[2]),angle)
            elements_plot[9:11] = [px/W,py/H]
            for i_e in range(len(elements)):
                if i_e == 0:
                    f.write("%d "%(elements[i_e]))
                elif i_e==(len(elements)-1):
                    f.write("%f\n"%(elements[i_e]))
                else:
                    f.write("%f "%(elements[i_e]))
            # print(elements_plot)
            img = utils.draw_rectangle(img,[elements_plot],colors=['cls','cls','cls','cls','yolored'],thickness=3,plot_kp="arrow",norm=True)
            # img = utils.draw_rectangle(img,[elements[:-2]],colors=['cls','cls','cls','cls','yolored'],thickness=3,plot_kp="arrow",norm=True)
        cv2.imwrite(os.path.join(args.plots,img_name),img)
        f.close()
            
            
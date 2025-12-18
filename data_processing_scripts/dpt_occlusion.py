import os
import argparse
import cv2
import numpy as np
import math
import utils
from skimage.draw import polygon2mask
import random

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--labels',required=True, default=None,
                    help='the directory to the labels')
parser.add_argument('--images', default=None,
                    help='the directory to the images')
parser.add_argument('--mode', default=None, type=str,
                    help='what type of occlusion do you want: black, white, blur, crop')
parser.add_argument('--chance', default=0.5, type=float,
                    help='what percentage of labels do you want to occlude')
parser.add_argument('--occlusion', default=0.5, type=float,
                    help='what percentage of the objects area do you want to occlude')
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

new_label_dir=os.path.join(args.labels, "occluded")
if not os.path.exists(new_label_dir):
    os.makedirs(new_label_dir)

new_image_dir=os.path.join(args.images, "occluded")
if not os.path.exists(new_image_dir):
    os.makedirs(new_image_dir)

new_mask_dir=os.path.join(args.images, "occluded_mask")
if not os.path.exists(new_mask_dir):
    os.makedirs(new_mask_dir)

def raw_labels(Labels, H, W):
    label_array = []
    for line in Labels:
        current_line = line[:-1]
        elements=current_line.split(" ")
        for i in range(len(elements)):
            if i==0:
                elements[i]=int(elements[i])
            elif i<9:
                if i%2:
                    elements[i]=int(float(elements[i])*W)
                else:
                    elements[i]=int(float(elements[i])*H)
        cosD=float(elements[9])
        sinD=float(elements[10])
        angleD=math.atan2(sinD,cosD)
        if angleD<0:
            angleD=math.pi*2+angleD
        rect = np.array((elements[1:9]))
        rect = cv2.minAreaRect(rect.reshape(4, 2))
        Dx,Dy=utils.dptFromAngle(rect[0][0],rect[0][1],rect[1][0],rect[1][1],math.radians(rect[2]),angleD)
        elements[9]=Dx
        elements[10]=Dy
        label_array.append(elements)
    return label_array

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
                img = cv2.imread(os.path.join(args.images, img_name))
        if img is None:
            print("Image %s does not exists"%(label_name[:-3]))
            continue
        height,width,_=img.shape
        labels_R = raw_labels(Lines,height,width)
        # mask = np.zeros((height,width))
        mask_dpts = np.zeros_like(img)
        mask_chosen = np.zeros_like(img)
        blurred_img = cv2.GaussianBlur(img, (21, 21), 0)
        for lindex in range(len(labels_R)):

            this_label = labels_R[lindex]
            ### from https://stackoverflow.com/a/36400130
            dpx=this_label[9]
            dpy=this_label[10]
            rect_pred_xywhr = cv2.minAreaRect(np.asarray(this_label)[1:9].reshape(4, 2))
            h_obb=rect_pred_xywhr[1][1]
            w_obb=rect_pred_xywhr[1][0]
            if h_obb>w_obb:
                h_obb=w_obb
            cx,cy = utils.get_center(this_label,integer=True)
            distance = math.sqrt((dpx-cx)**2+(dpy-cy)**2)
            angle = math.atan2(dpy-cy,dpx-cx)
            corners = utils.xywhr2xyxyxyxy(np.array([dpx-(distance*args.occlusion-distance/4)*math.cos(angle),dpy-(distance*args.occlusion-distance/4)*math.sin(angle),distance*2*args.occlusion+distance/2,h_obb,angle]))
            # corners = utils.xywhr2xyxyxyxy(np.array([dpx,dpy,distance*args.occlusion*4,distance/2,angle]))
            coordinates = [[y,x] for [x,y] in corners]
            polygon = np.array(coordinates)
            mask_bool = polygon2mask((height,width), polygon)         
            mask_dpts[mask_bool] = 255
            if random.random()<1-args.chance:
                continue
            mask_chosen[mask_bool] = 255
            if args.mode=="black":
                img[mask_bool] = 0
            if args.mode=="white":
                img[mask_bool] = 255
        
        cv2.imwrite(os.path.join(new_mask_dir,img_name),mask_chosen)
        if args.mode in ["black",'white']:            
            cv2.imwrite(os.path.join(new_image_dir,img_name),img)
            mask_chosen = cv2.cvtColor(mask_chosen,cv2.COLOR_BGR2GRAY)
            _,mask_chosen = cv2.threshold(mask_chosen,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        else:        
            if args.mode=="blur":
                out = np.where(mask_chosen==255, blurred_img, img)
            mask_chosen = cv2.cvtColor(mask_chosen,cv2.COLOR_BGR2GRAY)
            _,mask_chosen = cv2.threshold(mask_chosen,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            if args.mode=="crop":
                mask_dpts = cv2.cvtColor(mask_dpts,cv2.COLOR_BGR2GRAY)
                _,mask_dpts = cv2.threshold(mask_dpts,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                contours, hier = cv2.findContours(mask_chosen, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
                out = img.copy()
                for c in contours:            
                    x,y,w,h = cv2.boundingRect(c)            
                    search_empty_spot = 1
                    while search_empty_spot>0 and search_empty_spot<50:
                        ex1 = int(random.random()*(width-w))
                        ey1 = int(random.random()*(height-h))
                        ex2=ex1+w
                        ey2=ey1+h
                        if cv2.countNonZero(mask_dpts[ey1:ey2,ex1:ex2])<1:
                            search_empty_spot=0
                        else:
                            search_empty_spot+=1
                    donor = img[ey1:ey2,ex1:ex2]
                    target = img[y:y+h,x:x+w]
                    local_mask = mask_chosen[y:y+h,x:x+w]
                    local_mask = cv2.cvtColor(local_mask,cv2.COLOR_GRAY2BGR)
                    local_out = np.where(local_mask==255, donor, target)
                    out[y:y+h,x:x+w]=local_out

            cv2.imwrite(os.path.join(new_image_dir,img_name),out)

        savename_label = os.path.join(new_label_dir, label_name)
        f = open(savename_label, "w")
        for lindex in range(len(labels_R)):
            if mask_chosen[labels_R[lindex][10],labels_R[lindex][9]]:
                labels_R[lindex][11]=1
                occluded+=1
            else:
                visible+=1
            labels_R[lindex][1]=float(labels_R[lindex][1])/width
            labels_R[lindex][2]=float(labels_R[lindex][2])/height
            labels_R[lindex][3]=float(labels_R[lindex][3])/width
            labels_R[lindex][4]=float(labels_R[lindex][4])/height
            labels_R[lindex][5]=float(labels_R[lindex][5])/width
            labels_R[lindex][6]=float(labels_R[lindex][6])/height
            labels_R[lindex][7]=float(labels_R[lindex][7])/width
            labels_R[lindex][8]=float(labels_R[lindex][8])/height
            labels_R[lindex][9]=float(labels_R[lindex][9])/width
            labels_R[lindex][10]=float(labels_R[lindex][10])/height
            for eindex in range(len(labels_R[lindex])):
                if eindex==0:
                    f.write("%d "%(labels_R[lindex][eindex]))
                elif eindex==11:
                    f.write("%d\n"%(labels_R[lindex][eindex]))
                else:
                    f.write("%.16f "%(labels_R[lindex][eindex]))
        f.close
    print("Visible: %d"%(visible))
    print("Occluded: %d"%(occluded))
            
            
            
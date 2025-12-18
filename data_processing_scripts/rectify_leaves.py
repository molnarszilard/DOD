import cv2
import argparse
import numpy as np
import os
import utils
import math
import random

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Extract the individual leaves from an image.')
    parser.add_argument('--image', default="",type=str,
                        help='the directory to the source images (can end in jpg or png for only one file)')
    parser.add_argument('--contours', default="",type=str,
                        help='the directory to the segmentation contours')    
    parser.add_argument('--labels', default="",type=str,
                        help='the directory to the leaf labels')
    parser.add_argument('--leaf', default="",type=str,
                        help='the directory to the laboratory leaf images')
    parser.add_argument('--out', default="",type=str,
                        help='the directory in which to save the modified images')
    parser.add_argument('--collage', default="",type=str,
                        help='the directory to the collated images')
    parser.add_argument('--mincr', default=150,type=int,
                        help='minimum crop size')
    parser.add_argument('--ellipse_circle', default=False, action='store_true',
                    help='ellipse_circle true transforms both the lab and outdoor imsages to a circle, false will transform the outdoor leaf into the lab')

    args = parser.parse_args()
    return args

def read_contours(Contours, H, W):
    contours_array = []
    for line in Contours:
        current_line = line[:-1]
        contour_raw=current_line.split(" ")
        if len(contour_raw)<6:
            continue
        contour_points = []
        j=1
        while j < len(contour_raw):
            x,y = int(float(contour_raw[j])*W),int(float(contour_raw[j+1])*H)
            contour_points.append([x,y])
            j+=2
        # contours=np.asarray(contours)
        contour_points=np.array(contour_points)
        contours_array.append(contour_points)
    return contours_array

def raw_labels2ellipses(Labels, H, W):
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
        
        rect_ref = np.array((elements[1:9]))
        rect_ref_xywhr = cv2.minAreaRect(rect_ref.reshape(4, 2))  
        ellipse = np.array([elements[0],rect_ref_xywhr[0][0],rect_ref_xywhr[0][1],rect_ref_xywhr[1][0],rect_ref_xywhr[1][1],rect_ref_xywhr[2],float(elements[9]),float(elements[10])])
        label_array.append(ellipse)
    return label_array

def match(labels, contours):
    matched_contours = []
    matched_labels = []
    for this_label in labels:
        cx = this_label[1]
        cy = this_label[2]
        potential_matches = []
        potential_distances = []
        for potential_contour in contours:
            inlier = cv2.pointPolygonTest(potential_contour,(cx,cy),measureDist=True)
            if inlier>=0:
                potential_matches.append(potential_contour)
                potential_distances.append(inlier)
        potential_distances=np.array(potential_distances)
        if len(potential_matches)==1:
            matched_contours.append(potential_matches[0])
            matched_labels.append(this_label)
        elif len(potential_matches)>1:
            closest_index = np.argmax(potential_distances)
            matched_contours.append(potential_matches[closest_index])
            matched_labels.append(this_label)
    return matched_labels, matched_contours

def process_image(directory, filename, args, path_crop, path_seg, path_affine, path_rot,path_collate_lab):
    ### read images
    path = os.path.join(directory, filename)
    img = cv2.imread(path)
    H, W = img.shape[:2]
    f_label = open(os.path.join(args.labels, filename[:-3]+"txt"), "r")
    Labels = f_label.readlines()
    f_label.close()
    labels = raw_labels2ellipses(Labels, H, W)
    f_label = open(os.path.join(args.contours, filename[:-3]+"txt"), "r")
    Contours = f_label.readlines()
    f_label.close()
    contours = read_contours(Contours, H, W)
    matched_labels, matched_contours = match(labels, contours)
    print("Found %d matches."%(len(matched_labels)))
    for i in range(len(matched_labels)):
        this_label = matched_labels[i]
        this_contour = matched_contours[i]
        x,y,w,h = cv2.boundingRect(this_contour)
        brect_size = max(w,h)
        
        paddingX = int(brect_size/10)
        paddingY = int(brect_size/10)
        startX = max(0,x-paddingX)
        startY = max(0,y-paddingY)
        endX = min(W-1,x+brect_size)
        endY = min(H-1,y+brect_size)
        if min(endX-startX,endY-startY)<args.mincr:
            continue
        mask = np.zeros_like(img)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        try:
            cv2.drawContours(mask, [this_contour], -1, color=255, thickness=cv2.FILLED)
        except:
            print("Error in draw_contours on mask.")
            print("lines number")
            print(len(this_contour))  
        contoured_img = img.copy()
        try:
            cv2.drawContours(contoured_img, [this_contour], -1, utils.get_color('yolored'), thickness=2)
        except:
            print("Error in draw_contours on contoured img.")
            print("lines number")
            print(len(this_contour))
        
        
        ### Crop leaf from image
        cr_img = img[startY:endY,startX:endX]
        ### Read and process leaf image
        leaffile = random.choice([x for x in os.listdir(args.leaf) if os.path.isfile(os.path.join(args.leaf, x))])
        leafimg = cv2.imread(os.path.join(args.leaf,leaffile))
        try:
            leafimg = cv2.resize(leafimg, dsize=(cr_img.shape[1],cr_img.shape[0]), interpolation=cv2.INTER_NEAREST)
        except:
            print("error in resize leaf img")
        #### affine transform from ellipse to unit circle: 
        #### https://www.mathworks.com/matlabcentral/answers/35083-affine-transformation-that-takes-a-given-known-ellipse-and-maps-it-to-a-circle-with-diameter-equal#answer_44017
        #### https://math.stackexchange.com/a/310779

        leafmask = utils.get_mask(leafimg)
        lCx,lCy,llambda1,llambda2,langle = utils.get_ellipse_params_mask(leafmask,leafmask,False)
        
        ### Affine transform leaf ellipse->circle
        lQ1 = np.asarray([[math.cos(langle),-math.sin(langle)],[math.sin(langle),math.cos(langle)]])
        lS1 = np.asarray([[1,0],[0,llambda1/llambda2]])
        lC1=lQ1@lS1@lQ1.T
        lD1 = (np.eye(2)-lC1)@np.asarray([[lCx],[lCy]])
        lA1=np.zeros((3,3))
        lA1[0:2,0:2]=lC1
        lA1[0:2,2:3]=lD1
        lA1[2,2]=1
        ### Affine transform leaf circle to a unit circle in (0,0)
        lS2 = np.asarray([[1/llambda1,0],[0,1/llambda1]])
        lD2 = np.asarray([[-lCx/llambda1],[-lCy/llambda1]])
        lA2=np.zeros((3,3))
        lA2[0:2,0:2]=lS2
        lA2[0:2,2:3]=lD2
        lA2[2,2]=1        
        leafimg = cv2.ellipse(leafimg, (lCx,lCy), (llambda1,llambda2), math.degrees(langle), 0, 360, utils.get_color('black'), 6)
        leafimg = cv2.ellipse(leafimg, (lCx,lCy), (llambda1,llambda2), math.degrees(langle), 0, 360, utils.get_color('green'), 2)
        leaf_dx,leaf_dy=utils.dptFromAngle(lCx,lCy,llambda1*2,llambda2*2,langle,math.radians(-90))
        leafimg = cv2.arrowedLine(leafimg, (lCx,lCy), (int(leaf_dx),int(leaf_dy)), utils.get_color('black'), 6) # direction arrow
        leafimg = cv2.arrowedLine(leafimg, (lCx,lCy), (int(leaf_dx),int(leaf_dy)), utils.get_color('green'), 2) # direction arrow
        if args.ellipse_circle:
            transform_lab_leaf = lA1
            leafimg = cv2.warpAffine(leafimg, transform_lab_leaf[0:2,:], (leafimg.shape[1], leafimg.shape[0]))
        
        ### Crop the contoured image and the mask
        cr_contoured_img = contoured_img[startY:endY,startX:endX]
        cr_mask = mask[startY:endY,startX:endX]
        ### segment leaf
        cr_segmented_img = np.zeros_like(cr_img)
        cr_segmented_img[cr_mask>0] = cr_img[cr_mask>0]
        ### get ellipse parameters for the cropped leaf
        cr_cx = int(this_label[1]-startX)
        cr_cy = int(this_label[2]-startY)
        cr_w = int(this_label[3]/2)
        cr_h = int(this_label[4]/2)
        angle=math.radians(this_label[5])
        dir_angle = math.atan2(this_label[7],this_label[6])
        cr_dx,cr_dy=utils.dptFromAngle(cr_cx,cr_cy,cr_w*2,cr_h*2,angle,dir_angle)
        cr_dx = int(cr_dx)
        cr_dy = int(cr_dy)
        ### Draw the ellipse on the cropped leaf
        cr_img_ellipse_plot = cr_img.copy()
        cr_img_ellipse_plot = cv2.ellipse(cr_img_ellipse_plot, (cr_cx,cr_cy), (cr_w,cr_h), math.degrees(angle), 0, 360, utils.get_color('black'), 6)
        cr_img_ellipse_plot = cv2.ellipse(cr_img_ellipse_plot, (cr_cx,cr_cy), (cr_w,cr_h), math.degrees(angle), 0, 360, utils.get_color('green'), 2)

        Q1 = np.asarray([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
        S1 = np.asarray([[1/cr_w,0],[0,1/cr_h]])
        C1=Q1@S1@Q1.T
        D1 = (np.eye(2)-C1)@np.asarray([[cr_cx],[cr_cy]])-np.asarray([[cr_cx],[cr_cy]])
        A1=np.zeros((3,3))
        A1[0:2,0:2]=C1
        A1[0:2,2:3]=D1
        A1[2,2]=1

        ### Apply homography to leaf
        new_d_unitc = A1[0:2,0:2]@np.asarray([[cr_dx],[cr_dy]])+np.expand_dims(A1[0:2,-1],-1)
        new_d_ref_unitc = (lA2@lA1)[0:2,0:2]@np.asarray([[leaf_dx],[leaf_dy]])+np.expand_dims((lA2@lA1)[0:2,-1],-1)
        new_dangle_unitc = math.atan2(new_d_ref_unitc[1],new_d_ref_unitc[0])-math.atan2(new_d_unitc[1],new_d_unitc[0])
        R_unitc = np.asarray([[math.cos(new_dangle_unitc),-math.sin(new_dangle_unitc),0],[math.sin(new_dangle_unitc),math.cos(new_dangle_unitc),0],[0,0,1]])
        if args.ellipse_circle:
            trans_matrix = np.linalg.inv(lA2)@R_unitc@A1
        else:
            trans_matrix = np.linalg.inv(lA2@lA1)@R_unitc@A1
        new_d = trans_matrix[0:2,0:2]@np.asarray([[cr_dx],[cr_dy]])+np.expand_dims(trans_matrix[0:2,-1],-1)
        # trans_matrix = A1
        homography_leaf = cv2.warpAffine(cr_segmented_img, trans_matrix[0:2,:], (cr_segmented_img.shape[1], cr_segmented_img.shape[0]))
        homography_leaf = utils.delete_edges(homography_leaf,2)
        homography_leaf, _ = utils.filter_bw(homography_leaf)

        ### Apply homography to drawned leaf
        homography_leaf_drawing = cv2.warpAffine(cr_img_ellipse_plot, trans_matrix[0:2,:], (cr_img_ellipse_plot.shape[1], cr_img_ellipse_plot.shape[0]))
        
        homography_leaf_drawing = cv2.arrowedLine(homography_leaf_drawing, (lCx,lCy), (int(new_d[0]),int(new_d[1])), utils.get_color('black'), 6) # direction arrow
        homography_leaf_drawing = cv2.arrowedLine(homography_leaf_drawing, (lCx,lCy), (int(new_d[0]),int(new_d[1])), utils.get_color('green'), 2) # direction arrow
        
        ### Prepare large image for unsegmented processing
        cr_img_ellipse_plot = cr_img.copy()
        cr_img_ellipse_plot = cv2.ellipse(cr_img_ellipse_plot, (cr_cx,cr_cy), (cr_w,cr_h), math.degrees(angle), 0, 360, utils.get_color('black'), 6)
        cr_img_ellipse_plot = cv2.ellipse(cr_img_ellipse_plot, (cr_cx,cr_cy), (cr_w,cr_h), math.degrees(angle), 0, 360, utils.get_color('green'), 2)
        # cr_img_ellipse_plot = cv2.arrowedLine(cr_img_ellipse_plot, (cr_cx,cr_cy), (cr_dx,cr_dy), utils.get_color('black'), 5) # direction arrow
        cr_img_ellipse_plot = cv2.arrowedLine(cr_img_ellipse_plot, (cr_cx,cr_cy), (cr_dx,cr_dy), utils.get_color('black'), 6) # direction arrow
        cr_img_ellipse_plot = cv2.arrowedLine(cr_img_ellipse_plot, (cr_cx,cr_cy), (cr_dx,cr_dy), utils.get_color('green'), 2) # direction arrow

        ### Rotate leaf
        collate_img = np.hstack((cr_img, cr_segmented_img, homography_leaf))
        collate_draw = np.hstack((cr_contoured_img, cr_img_ellipse_plot,homography_leaf_drawing))
        collate = np.vstack((collate_img, collate_draw))
        newfilename = filename[:-4]+"_"+str(i)+"_collage.jpg"
        cv2.imwrite(os.path.join(args.collage,newfilename), collate)

        newfilename = filename[:-4]+"_"+str(i)+".jpg"
        cv2.imwrite(os.path.join(path_crop,newfilename), cr_img)
        cv2.imwrite(os.path.join(path_seg,newfilename), cr_segmented_img)
        cv2.imwrite(os.path.join(path_affine,newfilename), homography_leaf)
       
        collate_img = np.hstack((leafimg, cr_img_ellipse_plot, homography_leaf_drawing))
        newfilename = filename[:-4]+"_"+str(i)+"_collate_lab.jpg"
        cv2.imwrite(os.path.join(path_collate_lab,"all",newfilename), collate_img)
        newfilename = filename[:-4]+"_"+str(i)+"_collate_lab1.jpg"
        cv2.imwrite(os.path.join(path_collate_lab,newfilename), leafimg)
        newfilename = filename[:-4]+"_"+str(i)+"_collate_lab2.jpg"
        cv2.imwrite(os.path.join(path_collate_lab,newfilename), cr_img_ellipse_plot)
        newfilename = filename[:-4]+"_"+str(i)+"_collate_lab3.jpg"
        cv2.imwrite(os.path.join(path_collate_lab,newfilename), homography_leaf_drawing)
        # exit()
    
if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.out):
        os.makedirs(args.out)
        print("The new directory for saving images is created!")
    path_crop = os.path.join(args.out,"crop")
    if not os.path.exists(path_crop):
        os.makedirs(path_crop)
        print("The new directory for saving cropped images is created!")
    path_seg = os.path.join(args.out,"seg")
    if not os.path.exists(path_seg):
        os.makedirs(path_seg)
        print("The new directory for saving cropped images is created!")
    path_affine = os.path.join(args.out,"affine")
    if not os.path.exists(path_affine):
        os.makedirs(path_affine)
        print("The new directory for saving cropped images is created!")
    path_rot = os.path.join(args.out,"rot")
    if not os.path.exists(path_rot):
        os.makedirs(path_rot)
        print("The new directory for saving cropped images is created!")
    if args.collage is not None:
        if not os.path.exists(args.collage):
            os.makedirs(args.collage)
            print("The new directory for saving labels is created!")
    path_collate_lab = os.path.join(args.collage,"collate_lab")
    if not os.path.exists(path_collate_lab):
        os.makedirs(path_collate_lab)
        print("The new directory for saving collated lab images is created!")
    if not os.path.exists(os.path.join(path_collate_lab,"all")):
        os.makedirs(os.path.join(path_collate_lab,"all"))
        print("The new directory for saving collated lab images is created!")
    
    ### Evaluate only one image
    if args.image.endswith('.png') or args.image.endswith('.jpg'):
        directory, filename = os.path.split(args.image)
        if not os.path.exists(args.image):
            print("The file: "+args.image+" does not exists.")
            exit()
        process_image(directory, filename, args, path_crop, path_seg, path_affine, path_rot,path_collate_lab)
        
    ### Evaluate the images in a folder
    else:
        if os.path.isfile(args.image):
            print("The specified file: "+args.image+" is not an jpg or png image, nor a folder containing jpg or png images. If you want to evaluate videos, use eval_video.py or demo_video.py.")
            exit()
        if not os.path.exists(args.image):
            print("The folder: "+args.image+" does not exists.")
            exit()
        dlist=os.listdir(args.image)
        dlist.sort()
        for filename in dlist:
            if filename.endswith(".png") or filename.endswith(".jpg"):
                print("Extracting from: "+filename)
                process_image(args.image,filename, args, path_crop, path_seg, path_affine, path_rot,path_collate_lab)
folder_labels = "labels/";
folder_labels_out = "labels_oriented/";
folder_images = "images/";
label_files = dir(folder_labels);
images = dir(folder_images);

path_label = strcat(folder_labels,label_files(3).name);
labels = readtable(path_label);
labels = labels{:,:};

path_img = strcat(folder_images,images(3).name);
img = imread(path_img);
img_shape = size(img);
H = img_shape(1);
W = img_shape(2);
imshow(img);

% for i=1:length(labels)
%     hold on;
%     cx = labels(i,2)*W;
%     cy = labels(i,3)*H;
%     w = labels(i,4)*W;
%     h = labels(i,5)*H;
%     if length(labels(i))==6
%         angle = labels(i,6);
%     else
%         angle = 0;
%     end
%     x1 = cx-w/2;
%     y1 = cy-h/2;
%     rectangle('Position',[x1,y1,w,h],'EdgeColor','red')
% end
clear rect_plot
% for i=1:length(labels)
for i=1:2
    if i>1
        delete(rect_plot)
    end
    hold on;
    c = labels(i,1);
    x1 = labels(i,2)*W;
    y1 = labels(i,3)*H;
    x2 = labels(i,4)*W;
    y2 = labels(i,5)*H;
    x3 = labels(i,6)*W;
    y3 = labels(i,7)*H;
    x4 = labels(i,8)*W;
    y4 = labels(i,9)*H;
    x_rect = [x1,x2,x3,x4,x1];
    y_rect = [y1,y2,y3,y4,y1];
    rect = polyshape(x_rect,y_rect);
    [cx,cy] = centroid(rect);
    rect_plot = plot(rect,'EdgeColor','red','FaceAlpha',0.0);
    hold on;
    centroic_plot = plot(cx,cy,'r*');
    while 1
        [x,y] = ginput(1);
        % crd = get (gca, 'CurrentPoint');
        % y = crd(1,2);
        % x = crd(1,1);
        
        in = inpolygon(x,y,x_rect,y_rect);
        if in
            orientation_plot = plot(x,y,'m*');
            break;
        end
    end
    angle = rad2deg(atan2((cy-y)/2,(cx-x)/2));
    if angle<=0
        angle = angle+180;
    else
        angle = 180+angle;
    end
    angle
    labels(i,1) = angle;
end

path_label_out = strcat(folder_labels_out,label_files(3).name);
writematrix(labels,path_label_out,'Delimiter',' ');

% for i = 3:length(images)
%     path = strcat(folder_images,images(i).name);
%     img = imread(path);
%     imshow(img)
% end
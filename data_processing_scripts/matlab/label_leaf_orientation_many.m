folder_labels = "/media/rambo/norris_ssd1/datasets/vineyards/detect_healthy/train/temp/labels/";
folder_labels_out = "/media/rambo/norris_ssd1/datasets/vineyards/detect_healthy/train/temp/labels_oriented/";
folder_images = "/media/rambo/norris_ssd1/datasets/vineyards/detect_healthy/train/temp/images/";
label_files = dir(folder_labels);
label_files = [label_files(3:end)];
images = dir(folder_images);
images = [images(3:end)];

index = 1; % index 1 is the first element
inlier = false; % this forced the click to be inside the bounding box
alert_dot = true; % a big red dot will flash for 0.5 second to show the location of the BB

path_label = strcat(folder_labels,label_files(index).name);
labels = readtable(path_label);
labels = labels{:,:};

path_img = strcat(folder_images,images(index).name);
img = imread(path_img);
img_shape = size(img);
H = img_shape(1);
W = img_shape(2);
imshow(img);

clear rect_plot
clear rects
size_of_labels = size(labels);
number_of_labels = size_of_labels(1);
rects = repmat(polyshape, number_of_labels,1);
centroids = zeros(number_of_labels,2);
available = ones(number_of_labels,1);
for i=1:number_of_labels
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
    rect_plots(i)=rect_plot;
    hold on;
    centroic_plot = plot(cx,cy,'g*');
    centroids(i,:)=[cx,cy];
    rects(i)=rect;
end

i=1;
while i<=number_of_labels
    set(gcf,'name',strcat(images(index).name,":   ",num2str(number_of_labels),"/",num2str(i)),'numbertitle','off')
    % figure('Name',strcat(images(index).name,':   ',num2str(i),"/",num2str(number_of_labels)),'NumberTitle','off')
    % title(strcat(images(index).name,':... ',num2str(i),"/",num2str(number_of_labels)), 'Interpreter', 'none');
    not_in = true;
    while not_in
        [x,y,button] = ginput(1);
        if button==3
            go_back = true;
            break;
        else
            go_back = false;
        end
        j=1;
        while j<=length(rects) && not_in
            if available(j)
                in = inpolygon(x,y,rects(j).Vertices(:,1)',rects(j).Vertices(:,2)');
            else
                in = false;
            end
            if in
                orientation_plot = plot(x,y,'g*');
                not_in = false;
                available(j) = 0;
            else
                j=j+1;

            end
        end
    end
    cx = centroids(j,1);
    cy = centroids(j,2);
    angle = rad2deg(atan2((cy-y)/2,(cx-x)/2));
    if angle<=0
        angle = angle+180;
    else
        angle = 180+angle;
    end
    labels(j,1) = angle;
    delete(rect_plots(j))
    if ~go_back
        i=i+1;
    else
        if i==1
            i=1;
        else
            i=i-1;
        end
    end
end
close
path_label_out = strcat(folder_labels_out,label_files(index).name);
writematrix(labels,path_label_out,'Delimiter',' ');
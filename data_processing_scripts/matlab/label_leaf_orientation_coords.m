%%% Function to label the petioles of the leaves
%%% Left click marks the petioles
%%% Right click goes back one label
%%% Middle click makes the rectangle disappear for 1 second (use it if you
%%%     can't see the underlying leaf sufficiently well)
%%% If you click outside of the image by less than 20 pixels, the values
%%%     will be put on the edge
%%% If you click outside of the image by more than 20 pixels the value is 
%%%     saved (you can mark a incorrect label for later, by searching for
%%%     value outside of the tange [0-1])

folder_labels = "labels_sq/";
folder_labels_out = "labels_dpt/";
% folder_labels_petioles = "labels_d/";
folder_images = "images/";
label_files = dir(folder_labels);
label_files = [label_files(3:end)];
images = dir(folder_images);
images = [images(3:end)];

%%% Arguments
index = 53; % index 1 is the first element
inlier = false; % this forced the click to be inside the bounding box
alert_dot = true; % a big red dot will flash for 0.5 second to show the location of the BB
order = true; % this flag allows to organize the labels into an order depending on the distance between the labels

path_label = strcat(folder_labels,label_files(index).name);
labels = readtable(path_label);
labels = labels{:,:};

path_img = strcat(folder_images,images(index).name);
disp(path_img)
img = imread(path_img);
img_shape = size(img);
H = img_shape(1);
W = img_shape(2);
imshow(img);
set(gcf, 'Position', get(0, 'Screensize'));
size_of_labels = size(labels);
number_of_labels = size_of_labels(1);
petioles = zeros(number_of_labels,2);
if order
    label_order = order_labels(labels,W,H);
else
    label_order = 1:number_of_labels;
end
clear rect_plot
i = 1;
while i<=number_of_labels
    set(gcf,'name',strcat(images(index).name,":   ",num2str(number_of_labels),"/",num2str(i)),'numbertitle','off')
    % figure('Name',strcat(images(index).name,':   ',num2str(i),"/",num2str(number_of_labels)),'NumberTitle','off')
    % title(strcat(images(index).name,':... ',num2str(i),"/",num2str(number_of_labels)), 'Interpreter', 'none');
    label_index = label_order(i);
    hold on;
    c = labels(label_index,1);
    x1 = labels(label_index,2)*W;
    y1 = labels(label_index,3)*H;
    x2 = labels(label_index,4)*W;
    y2 = labels(label_index,5)*H;
    x3 = labels(label_index,6)*W;
    y3 = labels(label_index,7)*H;
    x4 = labels(label_index,8)*W;
    y4 = labels(label_index,9)*H;
    x_rect = [x1,x2,x3,x4,x1];
    y_rect = [y1,y2,y3,y4,y1];
    rect = polyshape(x_rect,y_rect);
    [cx,cy] = centroid(rect);
    rect_plot = plot(rect,'EdgeColor','red','FaceAlpha',0.0);
    if alert_dot
        % alert_plot = rectangle('Position',[x1,y1,200,200],'FaceColor','red','Curvature',[1 1]);
        % alert_plot = viscircles([cx,cy],150,'Color','r','FaceColor','r');
        alert_plot = plot(cx, cy,'.r', 'MarkerSize',150);
        if i==1
            pause(1);
        else
            pause(0.5);
        end
        delete(alert_plot);
    end
    % centroic_plot = plot(cx,cy,'g*');
    wait_for_click = true;
    go_back = false;
    while wait_for_click
        [x,y,button] = ginput(1);
        switch button
            case 1
                in = inpolygon(x,y,x_rect,y_rect);
                if inlier
                    if in
                        % orientation_plot = plot(x,y,'m*');
                        wait_for_click = false;
                    end
                else
                    if x<0 && x>-20
                        x=0;
                    end
                    if x>=W && x<W+19
                        x=W-1;
                    end
                    if y<0 && y>-20
                        y=0;
                    end
                    if y>=H && y<H+19
                        y=H-1;
                    end
                    % orientation_plot = plot(x,y,'g*');
                    wait_for_click=false;
                end
                % petioles(label_index,:)=[x/W,y/H];
                labels(label_index,10) = x/W;
                labels(label_index,11) = y/H;
            case 2
                delete(rect_plot);
                pause(1)
                rect_plot = plot(rect,'EdgeColor','red','FaceAlpha',0.0);
            case 3
                go_back = true;
                wait_for_click = false;  
        end        
    end
    delete(rect_plot);
    % TODO: currently the center is in the middle, and '+180' is required to 
    % maintain the trigonometric circle values
    % If you consider the petioles as the center, and the range of
    % (-180,180), then you can remove the '+180' from the equation
    
    
    if ~go_back
        % angle = rad2deg(atan2((cy-y)/2,(cx-x)/2))+180;    
        % labels(label_index,1) = angle;
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

if ~exist(folder_labels_out, 'dir')
   mkdir(folder_labels_out)
end
% if ~exist(folder_labels_petioles, 'dir')
%    mkdir(folder_labels_petioles)
% end
path_label_out = strcat(folder_labels_out,label_files(index).name);
writematrix(labels,path_label_out,'Delimiter',' ');
% path_label_petioles = strcat(folder_labels_petioles,label_files(index).name);
% writematrix(petioles,path_label_petioles,'Delimiter',' ');
disp(index)

function label_orders = order_labels(unordered_labels,W,H)
    size_of_labels = size(unordered_labels);
    number_of_labels = size_of_labels(1);
    label_orders = zeros(number_of_labels,1);
    used_indices = zeros(number_of_labels,1);
    cx1 = 0;
    cy1 = 0;
    for io1=1:number_of_labels
        distance = 10000;
        new_point_index = 1;
        for io2=1:number_of_labels
            if used_indices(io2)>0
                continue
            end
            x1 = unordered_labels(io2,2)*W;
            y1 = unordered_labels(io2,3)*H;
            x2 = unordered_labels(io2,4)*W;
            y2 = unordered_labels(io2,5)*H;
            x3 = unordered_labels(io2,6)*W;
            y3 = unordered_labels(io2,7)*H;
            x4 = unordered_labels(io2,8)*W;
            y4 = unordered_labels(io2,9)*H;
            x_rect = [x1,x2,x3,x4,x1];
            y_rect = [y1,y2,y3,y4,y1];
            rect = polyshape(x_rect,y_rect);
            [cx2,cy2] = centroid(rect);
            dist = sqrt( (cx1-cx2)^2 + (cy1-cy2)^2 );
            if dist<distance
                distance=dist;
                new_point_index=io2;
            end
        end
        label_orders(io1)=new_point_index;
        used_indices(new_point_index)=1;
        x1 = unordered_labels(new_point_index,2)*W;
        y1 = unordered_labels(new_point_index,3)*H;
        x2 = unordered_labels(new_point_index,4)*W;
        y2 = unordered_labels(new_point_index,5)*H;
        x3 = unordered_labels(new_point_index,6)*W;
        y3 = unordered_labels(new_point_index,7)*H;
        x4 = unordered_labels(new_point_index,8)*W;
        y4 = unordered_labels(new_point_index,9)*H;
        x_rect = [x1,x2,x3,x4,x1];
        y_rect = [y1,y2,y3,y4,y1];
        rect = polyshape(x_rect,y_rect);
        [cx1,cy1] = centroid(rect);
    end
end
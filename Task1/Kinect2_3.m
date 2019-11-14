clear
%% Read the data
remainPtCloud = pcread('Clouds/Cloud7.ply');
remainPtCloud = removeInvalidPoints(remainPtCloud);
Cloud = remainPtCloud;
%% Setting parameters
%Set the maximum point-to-plane distance (2cm) for plane fitting.
maxDistance = 0.02;
cerr = 20; %color error
%You can see here the color of the subject
color1 = [181 144 30];
color2 = [203 175 55]; 

%Here I put some intervals for the given colors to be detected
Rcolor1 = [color1(1)-cerr color1(1)+cerr];
Gcolor1 = [color1(2)-cerr color1(2)+cerr];
Bcolor1 = [color1(3)-cerr color1(3)+cerr];

Rcolor2 = [color2(1)-cerr color2(1)+cerr];
Gcolor2 = [color2(2)-cerr color2(2)+cerr];
Bcolor2 = [color2(3)-cerr color2(3)+cerr];

p = 0; %number of subject planes

meanx_0 = 0;
meany_0 = 0;
meanz_0 = 0;
%% Loop for obtaining points
roi = [remainPtCloud.XLimits; remainPtCloud.YLimits; remainPtCloud.ZLimits];
for l = 1:1
for i= 1:10% approximate number of iterations to find the planes
    %% Some rules for running
    rng('shuffle','v5uniform');
    if roi(1,1)-roi(1,2) == 0 || roi(2,1)-roi(2,2) == 0 || roi(3,1)-roi(3,2) == 0
        remainPtCloud = Cloud;
        roi = [remainPtCloud.XLimits; remainPtCloud.YLimits; remainPtCloud.ZLimits];
    elseif (size(roi) ~= [3 2]) 
        remainPtCloud = Cloud;
        roi = [remainPtCloud.XLimits; remainPtCloud.YLimits; remainPtCloud.ZLimits];
    end
    %The area where to find the plane
    
    sampleIndices = findPointsInROI(remainPtCloud,roi); %Find points in the given area
    A = size(sampleIndices);
    if mod(A(1), 3) ~= 0
        remainPtCloud = Cloud;
        roi = [remainPtCloud.XLimits; remainPtCloud.YLimits; remainPtCloud.ZLimits];
        sampleIndices = findPointsInROI(remainPtCloud,roi);
    end
    
    %% Plane Detection
    [model,inlierIndices,outlierIndices, plane, remainPtCloud] = PlaneDetect(remainPtCloud,...
                maxDistance, sampleIndices); %this function detects plane in the gotten points
    k = size(plane.Color);% number of points in plane
    n = 0; % number of points in the subject plane
    plus = 0; % the detector of the subject plane being found
    %% Find necessary (object colored) points in the planes
    % Arrays for storing points
    Ax = [];
    Ay = [];
    Az = [];
    for j = 1:k(1)
        pointc = plane.Color(j,:);
        if ((pointc(1)>= Rcolor1(1)) && (pointc(2)<= Rcolor1(2)) ...
                && (pointc(2)>= Gcolor1(1)) && (pointc(2)<= Gcolor1(2)) ...
                && (pointc(3)>= Bcolor1(1)) && (pointc(3)<= Bcolor1(2)))...
                || ((pointc(1)>= Rcolor2(1)) && (pointc(2)<= Rcolor2(2)) ...
                && (pointc(2)>= Gcolor2(1)) && (pointc(2)<= Gcolor2(2)) ...
                && (pointc(3)>= Bcolor2(1)) && (pointc(3)<= Bcolor2(2))) % True if the plane have the subject color inside
            n = n+1;
            
            x_1 = plane.Location(j,1);
            y_1 = plane.Location(j,2);
            z_1 = plane.Location(j,3);
            
            Ax(n) = x_1;
            Ay(n) = y_1;
            Az(n) = z_1;
            
            plus = 1;
        end
    end
%% Calculate coordinates' means of the planes
    if (plus == 1) & n>65
            max_x = max(Ax);
            max_y = max(Ay);
            max_z = max(Az);
            min_x = min(Ax);
            min_y = min(Ay);
            min_z = min(Az);
            
            meanx_1 = (max_x+min_x)/2;
            meany_1 = (max_y+min_y)/2;
            meanz_1 = (max_z+min_z)/2;
            
            p = p+1;
            x = ((p-1)/p)*meanx_0+(1/p)*meanx_1;
            y = ((p-1)/p)*meany_0+(1/p)*meany_1;
            z = ((p-1)/p)*meanz_0+(1/p)*meanz_1;

            meanx_0 = meanx_1;
            meany_0 = meany_1;
            meanz_0 = meanz_1;
    end
    
        
    
end
end
%% Plotting
if plus == 1
        figure(2)
        pcshow(Cloud)
        hold on
        pcshow([x y z], 'w', 'markersize', 2000)
        text(x-0.05,y-0.05,z-0.05,'Object center','Color','white','FontSize',8)
        title('3D Point Cloud with center of the Box (blue)');
          xlabel('X (m)');
          ylabel('Y (m)');
          zlabel('Z (m)');
else
     fprintf("Sorry, right planes weren't detected because of wrong initialization");
end
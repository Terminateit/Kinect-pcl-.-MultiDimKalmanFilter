function [model,inlierIndices,outlierIndices, plane, remainPtCloud] = PlaneDetect(remainPtCloud,...
            maxDistance, sampleIndices)
%PLANEDETECT Summary of this function goes here
%   Detailed explanation goes here
[model,inlierIndices,outlierIndices] = pcfitplane(remainPtCloud,...
            maxDistance,'SampleIndices',sampleIndices);
plane = select(remainPtCloud,inlierIndices);
remainPtCloud = select(remainPtCloud,outlierIndices);
end




load('Y:\IntraOp_Micro\NeuroPixel\Neuropixel\NeuropixelPaper\Code\Data\ChanMap.mat')
addpath(genpath('X:\Projects\Lab_Materials\Analysis_Tools_and_Software\ExampleCodeRunning\bipolar_colormap'))
%%
clf
xloc=Chanmap.xcoords;
yloc=Chanmap.ycoords;

t=1000000:1001000;
SEL=1:4:372;
v1 = LFPMatrix(SEL,t);
x1 = 1:93;
resolution = 0.05; %(1uM)
xq1 = 1:resolution:93;
vq1 = interp1(x1,v1,xq1,'spline');  
SEL=2:4:372;
v2 = LFPMatrix(SEL,t);
x2 = 1:93;
resolution = 0.05; %(1uM)
xq2 = 1:resolution:93;
vq2 = interp1(x2,v2,xq2,'spline'); 
SEL=3:4:372;
v3 = LFPMatrix(SEL,t);
x3 = 1:93;
resolution = 0.05; %(1uM)
xq3 = 1:resolution:93;
vq3 = interp1(x3,v3,xq3,'spline');    
SEL=4:4:372;
v4 = LFPMatrix(SEL,t);
x4 = 1:93;
resolution = 0.05; %(1uM)
xq4 = 1:resolution:93;
vq4 = interp1(x4,v4,xq4,'spline'); 

[xq,yq1] = meshgrid(11:1:59, 20:2:3700);
[xq,yq2] = meshgrid(11:1:59, 40:2:3720);
[xq,yq3] = meshgrid(11:1:59, 20:2:3700);
[xq,yq4] = meshgrid(11:1:59, 40:2:3720);

shading flat
colormap(bipolar2)

% scatter(repmat([11 27 43 59],size(vq1,1),1),...
%     [yq1(:,1) yq2(:,1) yq3(:,1) yq4(:,1)],...
%     [vq1(:,4) vq2(:,4) vq3(:,4) vq4(:,4)])
hold on
scatter(repmat([11 ],size(vq1,1),1),...
    [yq1(:,1) ],...
    120,[vq1(:,4)],'filled','marker','s')
scatter(repmat([ 27],size(vq1,1),1),...
    [yq2(:,1) ],...
    120,[vq2(:,4)],'filled','marker','s')
scatter(repmat([ 43 ],size(vq1,1),1),...
    [yq3(:,1) ],...
    120,[vq3(:,4)],'filled','marker','s')
scatter(repmat([59],size(vq1,1),1),...
    [yq4(:,1)],...
    120,[vq4(:,4)],'filled','marker','s')
caxis([-200 200])
shading flat
axis equal





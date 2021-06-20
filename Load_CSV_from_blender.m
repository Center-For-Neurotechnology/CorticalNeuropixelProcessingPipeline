function []=Load_CSV_from_blender(data_file, START_TIME, END_TIME)

% Once the blender output (the traced lines) are exported into .csv files, 
% below involves importing the .csv files back into matlab, 
% splicing the traces together, and calibrating the curves to fit the channels.

addpath(fullfile(pwd,'util'))

MAP = 2;    % should be either 2 or 4
COLUMN = 2; % select one column neuropixel column to process (1:MAP). 

FS = 2500;
DOWN_SAMPLE_FACTOR = 1; % change to 2,5 or 10 if output stl is too big



BLENDER_NORM_FACTOR = 0.07;

%% Load CSV file back:
data_path = ['enter data path for lf.bin'];  
csv_file = ['name of lf.bin file'];  

data_file = fullfile(data_path,data_file);
[LFPMatrix, meta] = read_LFP_from_bin(data_file, START_TIME, END_TIME);

time_range=1:DOWN_SAMPLE_FACTOR:length(LFPMatrix);
Z = (LFPMatrix(COLUMN:MAP:(384),time_range));

A=importdata(fullfile(data_path, csv_file));
B = sortrows(A,1);


BlenderCurveX = 0 + START_TIME + B(:,1) * DOWN_SAMPLE_FACTOR;
BlenderCurveY = BLENDER_NORM_FACTOR * abs(-B(:,2)) - 0;               

AssociatedFileDirectory=data_file;
save(fullfile(data_path, 'BlenderCurves.mat'),'BlenderCurveX','BlenderCurveY','AssociatedFileDirectory')

% pcolor(time_range,1:size(Z,1),Z)
% shading flat
% hold on
% plot(BlenderCurveX,BlenderCurveY,'color','c','linewidth',3)
% caxis([-200 200])
% xlim([time_range(floor(length(time_range)/2))-10000 time_range(floor(length(time_range)/2))+10000])





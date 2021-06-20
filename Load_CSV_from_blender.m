function [output_file]=Load_CSV_from_blender(csv_file, data_file, START_TIME, END_TIME)

% Once the blender output (the traced lines) are exported into .csv files, 
% below involves importing the .csv files back into matlab, 
% splicing the traces together, and calibrating the curves to fit the channels.

addpath(fullfile(pwd,'util'))

MAP = 2;    % should be either 2 or 4
COLUMN = 2; % select one column neuropixel column to process (1:MAP). 

FS = 2500;
DOWN_SAMPLE_FACTOR = 5; % change to 2,5 or 10 if output stl is too big (should be similar values as in Export_STL.m)

[data_path,binName,ext] = fileparts(data_file);

BLENDER_NORM_FACTOR = 0.07;

%% Load CSV file back:

[LFPMatrix, meta] = read_LFP_from_bin(data_file, START_TIME, END_TIME);

time_range=1:DOWN_SAMPLE_FACTOR:length(LFPMatrix);
Z = (LFPMatrix(COLUMN:MAP:(384),time_range));

A=importdata(csv_file);
B = sortrows(A,1);

BlenderCurveX = (0 + START_TIME + B(:,1) * DOWN_SAMPLE_FACTOR);
BlenderCurveY = BLENDER_NORM_FACTOR * abs(-B(:,2)) - 0;               

AssociatedFileDirectory=data_file;
output_file = fullfile(data_path, 'BlenderCurves.mat');
save(output_file,'BlenderCurveX','BlenderCurveY','AssociatedFileDirectory')

figure
pcolor(time_range,1:size(Z,1),Z)
shading flat
hold on
plot(BlenderCurveX,BlenderCurveY,'color','b','linewidth',3)
caxis([-200 200])





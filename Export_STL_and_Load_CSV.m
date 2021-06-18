%%   STL to Blender:
%
%   This script will load LFP data from SpikeGLX bin file and export it as stl file for manual annotation
%   Next, blender CSV output file is loaded back into matlab
%
%   June, 2021

clear all
addpath(fullfile(pwd,'util'))

MAP = 4;    % should be either 2 or 4
COLUMN = 2; % select one column neuropixel column to process (1:MAP). 

FS = 2500;
DOWN_SAMPLE_FACTOR = 1; % change to 2,5 or 10 if output stl is too big

data_path = ['enter data path'];                      % 'D:\Neuropixel\NeuropixelMG29\test5fffdfc_g0\test5fffdfc_g0_imec0\';
data_file = ['name of lf.bin file'];                  % 'test5fffdfc_g0_t0.imec0.lf.bin';

% select time range to load and export (in seconds):
START_TIME = ['enter start time here'] / FS;            %4.45e5 / FS;
END_TIME = ['enter end time here'];                     % 8.498e5 / FS;

%% load data from bin and export to STL: 
data_file = fullfile(data_path,data_file);
[LFPMatrix, meta] = read_LFP_from_bin(data_file, START_TIME, END_TIME);


time_range=1:DOWN_SAMPLE_FACTOR:length(LFPMatrix);

Z = (LFPMatrix(COLUMN:MAP:(384),time_range));
fvc = surf2patch(Z,'triangles') ;
stlwrite(fullfile(data_path,'LFPMatrix.stl'), fvc.faces, fvc.vertices)

%% Manual step in In Blender : 
% This section was to export the LFP into .stl files which can be imported
% as surface files into Blender (https://www.blender.org/)

%%%%%%%%%%%%%%%
% Importing the files into blender involves using the .stl Add-on (Import-Export: STL input) which 
% generates a blender object plane of channel x time  with the peaks and valleys (topography) the voltage of the signal.
% Within blender, to make the shifting sinks and peaks more visible, the short axis (channels) is expanded to 400 pixels 
% without changing the long axis of the surface (which is in time). 
% Then, based on the peaks and valleys of the distinctive LFP, we added a Stroke using GreasePencil to trace the movement artifact manually.
% Once the entire recording was traced using this tool, we converted the Stroke into a Line then into an Object and used custom python code 
% (exportingCSVinfoExample.py) to export the vertices of the traced Line into a .csv file
%%%%%%%%%%%%%%%

%% Load CSV 
% Once the blender output (the traced lines) are exported into .csv files, 
% below involves importing the .csv files back into matlab, 
% splicing the traces together, and calibrating the curves to fit the channels.

BLENDER_NORM_FACTOR = 0.07;

%% Load CSV file back:
data_path = ['enter data path'];  
csv_file = ['name of lf.bin file'];  
% csv_file = 'IntraopMG29Range2Part4.csv';

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





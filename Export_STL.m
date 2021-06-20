function []=Export_STL(data_file, START_TIME, END_TIME)
%   STL to Blender:
%
%   This script will load LFP data from SpikeGLX bin file and export it as stl file for manual annotation
%   Next, blender CSV output file is loaded back into matlab
%
% input:
%     data_file:    full path to ap.bin file
%     start_point:  start point in seconds
%     end_point:    end point in seconds

addpath(fullfile(pwd,'util'))

MAP = 2;    % should be either 2 or 4
COLUMN = 2; % select one column neuropixel column to process (1:MAP). 

FS = 2500;
DOWN_SAMPLE_FACTOR = 1; % change to 2,5 or 10 if output stl is too big


%% load data from bin and export to STL: 
[LFPMatrix, meta] = read_LFP_from_bin(data_file, START_TIME, END_TIME);

time_range=1:DOWN_SAMPLE_FACTOR:length(LFPMatrix);

Z = (LFPMatrix(COLUMN:MAP:(384),time_range));
fvc = surf2patch(Z,'triangles') ;
stlwrite(fullfile(data_path,'LFPMatrix.stl'), fvc.faces, fvc.vertices)



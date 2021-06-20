function [stl_file]=Export_STL(data_file, START_TIME, END_TIME, map, column)
%   STL to Blender:
%
%   This script will load LFP data from SpikeGLX bin file and export it as stl file for manual annotation
%   Next, blender CSV output file is loaded back into matlab
%
% input:
%     data_file:    full path to ap.bin file
%     start_point:  start point in seconds
%     end_point:    end point in seconds
%     map:          2 - analyze as two parallel columns, 4 - analyze as four parallel columns
%     column:       select one neuropixel column to process (1:MAP). 

addpath(fullfile(pwd,'util'))

% map = 2;    % should be either 2 or 4
% column = 2; % select one column neuropixel column to process (1:MAP). 

FS = 2500;
DOWN_SAMPLE_FACTOR = 5; % change to 2,5 or 10 if output stl is too big (should be similar values as in Load_CSV_from_blender.m)


%% load data from bin and export to STL: 
[LFPMatrix, meta] = read_LFP_from_bin(data_file, START_TIME, END_TIME);

time_range=1:DOWN_SAMPLE_FACTOR:length(LFPMatrix);

Z = (LFPMatrix(column:map:(384),time_range));
fvc = surf2patch(Z,'triangles') ;
stl_file = fullfile(fileparts(data_file),'LFPMatrix.stl');
stlwrite(stl_file, fvc.faces, fvc.vertices)



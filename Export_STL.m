function []=Export_STL(data_file, START_TIME, END_TIME)
%   STL to Blender:
%
%   This script will load LFP data from SpikeGLX bin file and export it as stl file for manual annotation
%   Next, blender CSV output file is loaded back into matlab
%
% input:
%     data_file:    full path to ap.bin file
%     channels:     a vector with all channels to include (1:384)
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

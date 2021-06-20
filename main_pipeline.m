% This is an example code that shows how to export and import back data for manual adjustment of human Neuropixels recordings 
% with Blender, following by adjusting the ap.bin data accroding to manual
% annotation.

lfp_data_file = '{your data path}\data_file.lfp.bin';
ap_data_file = '{your data path}\data_file.ap.bin';

START_TIME = 0; % seconds
END_TIME = 500; % seconds
name = 'Subject_name';
channels = 1:384;


%% Export to STL file:

stl_file = Export_STL(lfp_data_file, START_TIME, END_TIME);
disp (['load stl file: ',stl_file,' to Blender as surface']);

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
%
% exportingCSVinfoExample.py 
%
% to export the vertices of the traced Line into a .csv file which then is
% loaded back into matlab
%%%%%%%%%%%%%%%
%% Load data from CSV back to matlab:

[BlenderCurves_file] = Load_CSV_from_blender(lfp_data_file, START_TIME, END_TIME);

%% AP normalization:
start_point = 100;  % Seconds, take the stable recording time for further analysis 
end_point = 400;    % Seconds

[ap_normalized]=Neuropixel_AP_preProcessing(name, ap_data_file, channels, start_point, end_point);

%% AP alignment:

[bin_file_to_kilosort] = Intarpolate_and_align_AP(name, ap_normalized, BlenderCurves_file, channels, start_point, end_point);

%%%%%%%%%%%%%%
% The exported bin file can be sent to kilosort 3 for automatic sorting (https://github.com/MouseLand/Kilosort).
% next, the output clusters can be visualized and manually merge, split or discard in phy2 (https://phy.readthedocs.io/en/latest/).  
%
% waveforms matrics can then be exported with: 
% extract_waveform_with_metrics.py
% adapted from:
% (https://github.com/AllenInstitute/ecephys_spike_sorting/tree/master/ecephys_spike_sorting/modules/mean_waveforms)
% Source: Jia et al. (2019) "High-density extracellular probes reveal dendritic backpropagation and facilitate neuron classification." J Neurophys 121: 1831-1847





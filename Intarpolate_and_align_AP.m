function [output_file] = Intarpolate_and_align_AP(name, data_file, Blender_file, channels, start_point, end_point)

% input:
%     name:         string, name of subject
%     data_file:    full path to ap.bin file
%     Blender_file:    full path to mat file with blender output
%     channels:     a vector with all channels to include (1:384)
%     start_point:  start point in seconds
%     end_point:    end point in seconds

% no output - this will save a new bin file and additional information in
% a mat file

addpath(fullfile(pwd,'util'))

MAP = 2;    % should be either 2 or 4 (either use two columns or 4 columns)
LFP_FS = 2500;
AP_FS = 30000;

load(Blender_file,'BlenderCurveX','BlenderCurveY');

% read AP data:
data = read_AP_from_bin(data_file);

[filepath,binName,ext] = fileparts(data_file);
output_folder = [filepath,'\',name,'_aligned\'];
if ~exist(output_folder,'dir')
    mkdir(output_folder)
end


points(:,1) = BlenderCurveX;% * LFP_FS;
points(:,2) = BlenderCurveY;

% remove duplicate points if exist:
[~, ind] = unique(points(:,1));
duplicate_ind = setdiff(1:size(points,1),ind);
points(duplicate_ind,:)=[];

tt = [max(1,round(points(1,1))):round(points(end,1))]';

distortion = interp1(points(:,1),points(:,2),tt,'makima');

full_distortion = zeros(1,tt(end));
full_distortion(tt) = distortion;

full_distortion30 = repmat(full_distortion, AP_FS/LFP_FS, 1);
full_distortion30 = full_distortion30(1:end);

if start_point > 0
    distortion = - (full_distortion30(start_point*AP_FS:end_point*AP_FS) *(MAP/2));
else
    distortion = - (full_distortion30(1:end_point*AP_FS) *(MAP/2));
end

distortion = distortion - min(distortion)+1;

new_data = zeros(384 +ceil(max(distortion))*2,length(distortion),'single');

x = 1:192;
resolution = 0.05;
xq = 1:resolution:192;


for t = 1:length(distortion)
    rand_data = data(randperm(1e6,size(new_data,1)))';
    new_data(:,t) = rand_data;
    
    steps = floor(distortion(t));
    shift_steps = floor((distortion(t) - steps)/resolution);
    
    if rem(t,10000) == 0
        t
    end
    for map = 1:2
        x = channels(mod(channels,2) == mod(map,2));
        xq = linspace(x(1),x(end), (x(end)-x(1))/resolution +1);
        v = data(x,t);
        vq = interp1(x,v,xq,'spline');

        sampled_x = (x(1:end-1)-1)/resolution + shift_steps;
        if sampled_x(1) == 0
            sampled_x(1) = 1;
        end
        new_v = vq(sampled_x);
        new_ch =x(1:end-1)+steps*2;
        new_data(new_ch,t) = new_v';
    end
end

median_distortion = round(median(distortion)) *2;
output_file = [output_folder, binName ,'_To_kilosort.bin'];
fid = fopen(output_file,'w');
for t = 1:size(new_data,2)
    to_write = (new_data((1:384) + median_distortion,t)');
    fwrite( fid, int16(to_write), 'int16' );
    end
fclose(fid)

save([output_folder, binName ,'_To_kilosort','.mat'], 'median_distortion', 'start_point', 'end_point')
% save([output_folder, binName ,'_',num2str(start_point),'.mat'], 'data','meta','med','norm_factor','num_of_channels','-v7.3');
disp('Done!')





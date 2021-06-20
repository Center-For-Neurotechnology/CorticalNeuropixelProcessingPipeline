function [output_file]=Neuropixel_AP_preProcessing(name, data_file, channels, start_point, end_point)
% input:
%     name:         string, name of subject
%     data_file:    full path to ap.bin file
%     channels:     a vector with all channels to include (1:384)
%     start_point:  start point in seconds
%     end_point:    end point in seconds

% no output - this will save a new bin file and additional information in
% a mat file

% preferably use only stabe time
% dtrend each channel
% remove median values between channels.
% normalize channel by auto-covariance of data without outliers.

save_as_bin = true;

[filepath,binName,ext] = fileparts(data_file);
output_folder = [filepath,'\',name,'_normalized\'];
if ~exist(output_folder,'dir')
    mkdir(output_folder)
end


% read ap data:
[data, meta] = read_AP_from_bin(data_file, start_point, end_point);

% detrend data:
chanN = 384;
tic
for ch = channels
    disp(['detreanding #',num2str(ch)])
    data_array = detrend(data(ch,:));
    data(ch,:) = data_array;
end

% Remove median:
med =  median(data(channels,:), 1);
% save([output_folder, 'median.mat'], 'med')


for ch = channels 
    disp(['removing med #',num2str(ch)])
    data(ch,:) = data(ch,:) - med;
end

% Normalize channels:

for ch = channels 
    disp(['normilize #',num2str(ch)])
    outliers = isoutlier(data(ch,:),'quartiles');
    temp = std(data(ch,~outliers));
    norm_factor(ch) = (1 / temp)   *600;
    data(ch,:) = data(ch,:) * norm_factor(ch);
end


num_of_channels = length(channels);
output_file = [output_folder, binName ,'_',num2str(start_point),'.bin'];
fid = fopen(output_file,'w');
for t = 1:size(data,2)
    fwrite(fid,int16(data(channels,t)),'int16');
end
fclose(fid)

save([output_folder, binName ,'_',num2str(start_point),'.mat'], 'meta', 'med','norm_factor','num_of_channels', 'start_point', 'end_point')
% save([output_folder, binName ,'_',num2str(start_point),'.mat'], 'data','meta','med','norm_factor','num_of_channels','-v7.3');

disp('Done!');
end


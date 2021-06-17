
function [data, varargout] = read_LFP_from_bin(data_file, varargin)

FS = 2500;

if nargin>1
    start_point = varargin{1};
else
    start_point = false;
end


if nargin>2
    end_point = varargin{2};
else
    end_point = false;
end

[filepath,binName,~] = fileparts(data_file);

try % if meta file exist (spikeGLX recording)
    [meta] = ReadMeta(data_file, filepath);
catch
    disp('meta file was not found!')
    meta = [];
end

fid = fopen(data_file,'r');
if start_point
    start_in_file = start_point*FS*385*2;
    fseek(fid,start_in_file,'bof');
else
    start_point=0;
end

if end_point
    data = fread (fid,[385,(end_point - start_point)*FS],'int16=>single');
else
    data = fread (fid,[385,Inf],'int16=>single');
end
fclose(fid);
varargout{1} = meta;
end



function [meta] = ReadMeta(binName, path)

    % Create the matching metafile name
    [dumPath,name,dumExt] = fileparts(binName);
    metaName = strcat(name, '.meta');

    % Parse ini file into cell entries C{1}{i} = C{2}{i}
    fid = fopen(fullfile(path, metaName), 'r');
% -------------------------------------------------------------
%    Need 'BufSize' adjustment for MATLAB earlier than 2014
%    C = textscan(fid, '%[^=] = %[^\r\n]', 'BufSize', 32768);
    C = textscan(fid, '%[^=] = %[^\r\n]');
% -------------------------------------------------------------
    fclose(fid);

    % New empty struct
    meta = struct();

    % Convert each cell entry into a struct entry
    for i = 1:length(C{1})
        tag = C{1}{i};
        if tag(1) == '~'
            % remake tag excluding first character
            tag = sprintf('%s', tag(2:end));
        end
        meta = setfield(meta, tag, C{2}{i});
    end
end % ReadMeta

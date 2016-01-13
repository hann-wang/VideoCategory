%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script helps train object data
% based on previously trained data.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load('../dataset/TRAINset.mat');
total=length(TRAINclass);
%load previously trained data
load('./data/finalobj.mat');

for i=6:20
    %jump through bad videos
    if (data(i,1763)==0)
        continue;
    end
    if (data(i,1762)~=0)
        error('Trained entry found!');
    end
    disp(i);
    
    classstd    =   TRAINclass(i);
    filename    =   ['../dataset/train/',num2str(classstd),'/',TRAINfiles{i}];

    % load sample video data and full audio data
    [dat_vid, ~]    =   mmread(filename, [1 , 2], [], false, true);
    %[~, dat_aud]    =   mmread(filename, [], [], true, false);
    dat_vid.filename        =   filename;

    %% Object detection
    [maxface,faceratio,totalface,blobcnt,blobrate,cornercnt]=processObject(dat_vid.width,dat_vid.height,dat_vid.nrFramesTotal,false,dat_vid.rate,dat_vid.totalDuration,dat_vid.filename,false);
    
    %% Form data matrix
    data(i,1757:1762)=[maxface,faceratio,totalface,blobcnt,blobrate,cornercnt];
    save('./data/finalobj.mat','data');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script generates training data for 
% classification learner coming with Matlab R2015b
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function []=fun_datagen(start,stop)

    load('../dataset/TRAINset.mat');
    total=length(TRAINclass);
    data=zeros(total,1998);
    data(:,end)=TRAINclass;
    datafilename=['./data/data',num2str(start),'-',num2str(stop),'.mat'];
    for i=start:stop
        disp(i);

        classstd    =   TRAINclass(i);
        filename    =   ['../dataset/train/',num2str(classstd),'/',TRAINfiles{i}];

        % load sample video data and full audio data
        [dat_vid, ~]    =   mmread(filename, [1 , 2], [], false, true);
        %[~, dat_aud]    =   mmread(filename, [], [], true, false);
        dat_vid.filename        =   filename;
        
        %% Audio Characteristics
        [SP, DSP, VDSP, LFP, CP, ED, SCP, ZCR, RMS] = processAudio( dat_aud );
        
        %% Video Characteristics
        [rhythm,HotAction,ColdAction,GradualTrans,Hgw,He,Pcold,Pdark,Ppale,Padj,Pcom]=processVideo(dat_vid.width,dat_vid.height,dat_vid.rate,dat_vid.totalDuration,dat_vid.filename,dat_vid.nrFramesTotal,false,false);

        %% Form data matrix
        data(i,1:1762)=[SP, DSP, VDSP, LFP, CP, ED, SCP, ZCR, RMS];
        data(i,1763:1766)=[rhythm,HotAction,ColdAction,GradualTrans];
        data(i,1767:1982)=Hgw(:)';
        data(i,1983:1992)=He;
        data(i,1993:1997)=[Pcold,Pdark,Ppale,Padj,Pcom];
        save(datafilename,'data');
    end

    %record filename in datafilelist
    if (exist('./data/datafilelist.mat', 'file'))
        load('./data/datafilelist.mat');
        len=length(datafilelist);
        datafilelist{len+1}=datafilename;
        save('./data/datafilelist.mat','datafilelist');
    else
        datafilelist=cell(1,1);
        datafilelist{1}=datafilename;
        save('./data/datafilelist.mat','datafilelist');
    end
end

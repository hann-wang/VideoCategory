% process a video and return its genre using audio-visual information
%
% INPUT
%   dat_vid Video structure returned by mmread, additional field filename added for direct video file location
%   dat_aud Audio structure returned by mmread
%   img_dir Directory of images converted from the input video in time order
%   prev_rst    Standard genre result of the last call of this function
%
% OUTPUT
%   group_rst   A single integer indicating the group id.
%

%% edit your own code in this file, leave the function interface unmodified

function    group_rst   =   fun_process(dat_vid, dat_aud, img_dir, prev_rst)
    global trainedClassifier Hscatter CPscatter LFPscatter VDSPscatter DSPscatter SPscatter;
    global age prev_pct SP DSP VDSP LFP CP Hgw;
    if (isempty(trainedClassifier))
        load('./data/baggedtree.mat');
        load('./data/auxilary.mat');
    end
    
    %% Update scatter coordinates
    if (prev_rst>0)
        switch(prev_rst)
            case 1014
                t=1;
            case 1016
                t=2;
            case 1017
                t=3;
            case 1019
                t=4;
            case 1020
                t=5;
        end
        switch(prev_pct)
            case 1014
                p=1;
            case 1016
                p=2;
            case 1017
                p=3;
            case 1019
                p=4;
            case 1020
                p=5;
        end
        %Error vector towards true scatter
        Herr=Hgw(:)'-Hscatter(t,:);
        CPerr=CP-CPscatter(t,:);
        LFPerr=LFP-LFPscatter(t,:);
        VDSPerr=VDSP-VDSPscatter(t,:);
        DSPerr=DSP-DSPscatter(t,:);
        SPerr=SP-SPscatter(t,:);
        %aging
        age(t)=age(t)+1;
        Hscatter(t,:)=Hscatter(t,:)+Herr/age(t);
        CPscatter(t,:)=CPscatter(t,:)+CPerr/age(t);
        LFPscatter(t,:)=LFPscatter(t,:)+LFPerr/age(t);
        VDSPscatter(t,:)=VDSPscatter(t,:)+VDSPerr/age(t);
        DSPscatter(t,:)=DSPscatter(t,:)+DSPerr/age(t);
        SPscatter(t,:)=SPscatter(t,:)+SPerr/age(t);
        %punishment
        if (t~=p)
            %Error vector towards false scatter
            Herr=Hgw(:)'-Hscatter(p,:);
            CPerr=CP-CPscatter(p,:);
            LFPerr=LFP-LFPscatter(p,:);
            VDSPerr=VDSP-VDSPscatter(p,:);
            DSPerr=DSP-DSPscatter(p,:);
            SPerr=SP-SPscatter(p,:);
            age(p)=age(p)+1;
            Hscatter(p,:)=Hscatter(p,:)-Herr/age(p);
            CPscatter(p,:)=CPscatter(p,:)-CPerr/age(p);
            LFPscatter(p,:)=LFPscatter(p,:)-LFPerr/age(p);
            VDSPscatter(p,:)=VDSPscatter(p,:)-VDSPerr/age(p);
            DSPscatter(p,:)=DSPscatter(p,:)-DSPerr/age(p);
            SPscatter(p,:)=SPscatter(p,:)-SPerr/age(p);
        end
    end
    
    %% Audio Characteristics
    [SP,DSP,VDSP,LFP,CP,ED,SCP,ZCR,RMS] = processAudio( dat_aud );
    
    %% Object detection
    [maxface,faceratio,totalface,blobcnt,blobrate,cornercnt]=processObject(dat_vid.width,dat_vid.height,dat_vid.nrFramesTotal,img_dir,dat_vid.rate,dat_vid.totalDuration,dat_vid.filename,true);

    %% Video Characteristics
    [rhythm,HotAction,ColdAction,GradualTrans,Hgw,He,Pcold,Pdark,Ppale,Padj,Pcom]=processVideo(dat_vid.width,dat_vid.height,dat_vid.rate,dat_vid.totalDuration,dat_vid.filename,dat_vid.nrFramesTotal,img_dir,true);
    
    %% Scatter distance
    Hdist=zeros(1,5);
    CPdist=zeros(1,5);
    LFPdist=zeros(1,5);
    VDSPdist=zeros(1,5);
    DSPdist=zeros(1,5);
    SPdist=zeros(1,5);
    for j=1:5
        Hdist(j)=sqrt(sum((Hgw(:)'-Hscatter(j,:)).^2));
    end
    %CP distance
    for j=1:5
        CPdist(j)=sqrt(sum((CP-CPscatter(j,:)).^2));
    end
    %LFP distance
    for j=1:5
        LFPdist(j)=sqrt(sum((LFP-LFPscatter(j,:)).^2));
    end
    %VDSP distance
    for j=1:5
        VDSPdist(j)=sqrt(sum((VDSP-VDSPscatter(j,:)).^2));
    end
    %DSP distance
    for j=1:5
        DSPdist(j)=sqrt(sum((DSP-DSPscatter(j,:)).^2));
    end
    %SP distance
    for j=1:5
        SPdist(j)=sqrt(sum((SP-SPscatter(j,:)).^2));
    end
    
    %% Color Entropy
    color=Hgw(:)/100;
    color(color==0)=[];
    colorEntropy=-sum(color.*log2(color));
    
    
    %% machine learning model
    group_rst=trainedClassifier.predictFcn([ED, SCP, ZCR, RMS, SPdist,DSPdist,VDSPdist,LFPdist,CPdist,colorEntropy,Hdist,maxface,faceratio,totalface,blobcnt,blobrate,cornercnt,rhythm,HotAction,ColdAction,GradualTrans,He,Pcold,Pdark,Ppale,Padj,Pcom]);
    %group_rst=trainedClassifier.predictFcn([colorEntropy,Hdist,maxface,faceratio,totalface,blobcnt,blobrate,cornercnt,rhythm,HotAction,ColdAction,GradualTrans,He,Pcold,Pdark,Ppale,Padj,Pcom]);
    prev_pct=group_rst;
end

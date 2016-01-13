function [rhythm,HotAction,ColdAction,GradualTrans,Hgw,He,Pcold,Pdark,Ppale,Padj,Pcom] = processVideo(width,height,vidrate,totalDuration,filename,nrFramesTotal,img_dir,valid)
%Video Characteristics
%Temporal: rhythm, action, gradual transition ratio
%Color: Global Weighted Color Histogram, Elementary Color Histogram
    

    %% Read Video data
    framestep=max(1,round(vidrate/10));
    equalrate=vidrate/framestep;
    %max length to deal with: 10min
    if (totalDuration<400)
        if (valid)
            frames=[1:framestep:nrFramesTotal];
        else
            frames=[1:framestep:floor((floor(totalDuration)-1.5)*vidrate)];
        end
    else
        frames=[round((totalDuration/2-150)*vidrate):framestep:round((totalDuration/2+150)*vidrate)];
    end
    %number of frames;
    NumFrames=length(frames);
    %read maximum 100 frames at a time
    %and compress them
    compressed=cell(1,NumFrames);
    if (width>300)
        swidth=250;
        sheight=round(height/width*250);
        framesize=sheight*250;
        needcompress=true;
    else
        framesize=width*height;
        needcompress=false;
    end
    if (valid)
        %read images directly
        for i=1:NumFrames
            if (needcompress)
                compressed{i}=cv.resize(imread([img_dir, num2str(frames(i), '%06d.jpg')]),[swidth,sheight]);
            else
                compressed{i}=imread([img_dir, num2str(frames(i), '%06d.jpg')]);
            end
        end
    else
        for i=1:100:NumFrames
            %read frames from i to stoppos
            stoppos=min(i+99,NumFrames);
            [video,~] = mmread(filename, frames(i:stoppos), [], false, true);
            if (needcompress)
                for j=1:stoppos-i+1
                    %compressed{i-1+j}=imresize(video.frames(j).cdata,[sheight,swidth]);
                    compressed{i-1+j}=cv.resize(video.frames(j).cdata, [swidth,sheight]);
                end
            else
                for j=1:stoppos-i+1
                    compressed{i-1+j}=video.frames(j).cdata;
                end
            end
        end
    end
    
    %% Fades detection settings
    th_sigmafade=0.006;
    FADEmin=equalrate/4;
    fadingin=false;
    fadecnt=0;
    fadetime=zeros(1000,2);
    %brightness
    sigma=zeros(1,NumFrames);

    %% Cuts detection settings
    ECRmin=0.22;
    thECR=0.5;
    cuts=false(1,NumFrames);
    %Edge change rate
    ECR=zeros(1,NumFrames);
    
    %% deal with the 1st frame
    %img_hsv=rgb2hsv(compressed{1});
    img_hsv=cv.cvtColor(compressed{1}, 'RGB2HSV');
    sigma(1)=sum(sum((img_hsv(:,:,3)-mean(mean(img_hsv(:,:,3)))).^2))/framesize;
    if (sigma(1)<th_sigmafade)
        fadingin=true;
        fadecnt=fadecnt+1;
        fadetime(fadecnt,1)=1;
    end
    %find edges
    edge_old=cv.Canny(compressed{1}(:,:,3), 128);
    edgecnt_old=length(find(edge_old>0));

    %% deal with the rest frames
    for i = 2 : NumFrames
        %img_hsv=rgb2hsv(compressed{i});
        img_hsv=cv.cvtColor(compressed{i}, 'RGB2HSV');
        sigma(i)=sum(sum((img_hsv(:,:,3)-mean(mean(img_hsv(:,:,3)))).^2))/framesize;
        if (fadingin)
            %Entering the brighten process
            if (sigma(i)>sigma(i-1))
                if (i>2)
                    %judging with the increasing speed of brightness
                    if (sigma(i)-sigma(i-1)<0.1*(sigma(i-1)-sigma(i-2)))
                        if (i-fadetime(fadecnt,1)>FADEmin)
                            fadetime(fadecnt,2)=i-1;
                        else
                            fadecnt=fadecnt-1;
                        end
                        fadingin=false;
                    end
                else
                    %the increasing speed of brightness cannot be assessed
                    %at the first 2 frames
                    if (sigma(i)<sigma(i-1))
                        fadetime(fadecnt,2)=i-1;
                        fadingin=false;
                    end
                end
            end
            continue;
        else
            if (sigma(i)<th_sigmafade && i>1 && i<NumFrames-1)
                if (fadecnt>0 && i-fadetime(fadecnt,2)<0.25*equalrate)
                    %joint the last fading process
                    fadingin=true;
                else
                    cuts(i)=true;
                    fadingin=true;
                    fadecnt=fadecnt+1;
                    %detect the rise process
                    for j=i-1:-1:1
                        %%judging with the decreasing speed of brightness
                        if (sigma(j)-sigma(j+1)<0.1*(sigma(j+1)-sigma(j+2)))
                            fadetime(fadecnt,1)=j+1;
                            break;
                        end
                    end
                end
            end
        end
        %find edges
        edge_new=cv.Canny(compressed{i}(:,:,3), 128);
        edgecnt_new=length(find(edge_new>0));
        edge_old=edge_new-edge_old;
        %Edge Change Rate
        ECR(i)=max(length(find(edge_old>0))/edgecnt_new,length(find(edge_old<0))/edgecnt_old);
        edge_old=edge_new;
        edgecnt_old=edgecnt_new;
        %judge cuts
        if (ECR(i)<ECRmin)
            ECR(i)=0;
        end
    end
    if (fadingin)
        fadetime(fadecnt,2)=NumFrames;
    end

    %% Dissolve dectection
    DISmin=equalrate/2;
    dissolving=false;
    distime=zeros(1000,2);
    discnt=0;
    sigma_sd=diff(diff(sigma));
    for i=1:length(sigma_sd)
        if (dissolving)
            if (sigma_sd(i)<=0)
                dissolving=false;
                if (i-distime(discnt+1,1)>DISmin)
                    discnt=discnt+1;
                    distime(discnt,2)=i;
                end
            end
        else
            if (sigma_sd(i)>0)
                distime(discnt+1,1)=i;
                dissolving=true;
            end
        end
    end
    if (dissolving)
        discnt=discnt+1;
        distime(discnt,2)=NumFrames;
    end
    for i=1:discnt
        if (i>discnt)
            break;
        end
        cutpos=round((distime(i,1)+distime(i,2))/2);
        if (isempty(find(cuts(max(1,round(cutpos-equalrate/2)):min(NumFrames,round(cutpos+equalrate/2))),1)))
            cuts(cutpos)=true;
        else
            %delete an existing fading process
            distime(i,:)=[];
            discnt=discnt-1;
        end
    end
    
    %judge cuts
    [~,pos]=findpeaks(ECR,'minpeakheight',thECR,'minpeakdistance',equalrate,'MinPeakProminence',0.88);
    for i=1:length(pos)
        if (pos(i)>equalrate && pos(i) < NumFrames-equalrate)
            if (isempty(find(cuts(round(pos(i)-equalrate/2):round(pos(i)+equalrate/2)),1)))
                cuts(pos(i))=true;
            end
        end
    end
    
    %% Cut List
    cutlist=find(cuts);
    
    %% Calculate rhythm and Action
    %threshold of the Hot/Cold shot length
    thHot=3*equalrate;
    thCold=8.3*equalrate;
    shotcnt=length(cutlist)+1;
    shotlength=zeros(1,shotcnt);
    %deal with the 1st and last shots
    if (shotcnt==1)
        shotlength=NumFrames;
        HotAction=0;
        ColdAction=1;
    else
        shotlength(1)=cutlist(1);
        shotlength(end)=NumFrames-cutlist(end);
        HotAction=0;
        ColdAction=0;
        if (shotlength(1)>=thCold)
            ColdAction=ColdAction+1;
        else
            if (shotlength(1)<=thHot)
                HotAction=HotAction+1;
            end
        end
        if (shotlength(end)>=thCold)
            ColdAction=ColdAction+1;
        else
            if (shotlength(end)<=thHot)
                HotAction=HotAction+1;
            end
        end
        %deal with the rest shots
        for i=2:shotcnt-1
            shotlength(i)=cutlist(i)-cutlist(i-1);
            if (shotlength(i)>=thCold)
                ColdAction=ColdAction+1;
            else
                if (shotlength(i)<=thHot)
                    HotAction=HotAction+1;
                end
            end
        end
    end
    
    %rhythm=sum(shotlength)/shotcnt/equalrate;
    if (shotcnt>2)
        rhythm=mean(shotlength(1:end-1))/equalrate;
    else
        rhythm=mean(shotlength)/equalrate;
    end
    HotAction=HotAction/shotcnt;
    ColdAction=ColdAction/shotcnt;
    
    %% Gradual Transition Ratio
    GradualTrans=0;
    for i=1:fadecnt
        GradualTrans=GradualTrans+fadetime(i,2)-fadetime(i,1)+1;
    end
    for i=1:discnt
        GradualTrans=GradualTrans+distime(i,2)-distime(i,1)+1;
    end
    GradualTrans=GradualTrans/NumFrames;
    
    %% Global Weighted Color Historam
    Hgw=zeros(6,6,6);
    pos=shotlength(1);
    for i=1:shotcnt
        if (i~=1)
            imgpos=round(pos+shotlength(i)/2);
            pos=pos+shotlength(i);
        else
            imgpos=round(shotlength(1)/2);
        end
        Hnew=cv.calcHist({compressed{imgpos}(:,:,1), compressed{imgpos}(:,:,2), compressed{imgpos}(:,:,3)}, {[0,255], [0,255], [0,255]}, 'HistSize',[6,6,6], 'Uniform',true);
        Hgw=Hgw+Hnew/framesize*shotlength(i)/NumFrames*100;
    end
    
    %% Elementary Color Histogram
    %1~10; Red	Pink	Orange	Yellow	Purple	Green	Blue	Brown	White	Gray
    He=zeros(1,10);
    He(1)=Hgw(6, 1 , 1 )+Hgw(6, 4 , 4 )+Hgw(5, 3 , 3 )+Hgw(5, 1 , 2 )+Hgw(4, 1 , 1 )+Hgw(2, 1 , 1)+Hgw(3, 1 , 1)+Hgw(5, 1 , 1)+Hgw(4, 1 , 2)+Hgw(5, 1 , 3)+Hgw(5, 2 , 1)+Hgw(5, 2 , 2)+Hgw(5, 2 , 3)+Hgw(6, 1 , 2)+Hgw(6, 1 , 3)+Hgw(6, 2 , 2)+Hgw(6, 2 , 3);
    He(2)=Hgw(6, 3 , 5)+Hgw(6, 1 , 4)+Hgw(5, 3 , 4)+Hgw(5, 1 , 4)+Hgw(5, 1 , 5)+Hgw(5, 1 , 6)+Hgw(5, 2 , 4)+Hgw(6, 1 , 5)+Hgw(6, 2 , 4)+Hgw(6, 2 , 5)+Hgw(6, 2 , 6)+Hgw(6, 3 , 4)+Hgw(6, 3 , 6)+Hgw(6, 4 , 5);
    He(3)=Hgw(6, 4 , 1)+Hgw(6, 3 , 3)+Hgw(6, 3 , 2)+Hgw(6, 2 , 1)+Hgw(6, 3 , 1)+Hgw(6, 4 , 2)+Hgw(6, 5 , 2)+Hgw(6, 5 , 3);
    He(4)=Hgw(6, 6 , 1)+Hgw(6, 6 , 4)+Hgw(5, 5 , 3)+Hgw(6, 5 , 1)+Hgw(4, 3 , 1)+Hgw(5, 6 , 1)+Hgw(5, 6 , 2)+Hgw(5, 6 , 3)+Hgw(6, 6 , 2)+Hgw(6, 6 , 3);
    He(5)=Hgw(4, 1 , 4)+Hgw(5, 4 , 5)+Hgw(6, 4 , 6)+Hgw(5, 3 , 5)+Hgw(6, 1 , 6)+Hgw(4, 3 , 5)+Hgw(4, 2 , 5)+Hgw(4, 1 , 5)+Hgw(3, 3 , 6)+Hgw(3, 3 , 5)+Hgw(2, 2 , 4)+Hgw(2, 1 , 4)+Hgw(2, 1 , 2)+Hgw(2, 1 , 3)+Hgw(2, 2 , 3)+Hgw(3, 1 , 2)+Hgw(3, 1 , 3)+Hgw(3, 1 , 4)+Hgw(3, 1 , 5)+Hgw(3, 1 , 6)+Hgw(3, 2 , 3)+Hgw(3, 2 , 4)+Hgw(3, 2 , 5)+Hgw(3, 2 , 6)+Hgw(4, 1 , 3)+Hgw(4, 1 , 6)+Hgw(4, 2 , 3)+Hgw(4, 2 , 4)+Hgw(4, 2 , 6)+Hgw(4, 3 , 4)+Hgw(4, 3 , 6)+Hgw(5, 2 , 5)+Hgw(5, 2 , 6)+Hgw(5, 3 , 6)+Hgw(5, 4 , 6);
    He(6)=Hgw(1, 4 , 1 )+Hgw(4, 6 , 4 )+Hgw(4, 5 , 2 )+Hgw(4, 6 , 2 )+Hgw(3, 6 , 1 )+Hgw(1, 6 , 1 )+Hgw(2, 5 , 2 )+Hgw(1, 6 , 4 )+Hgw(1, 6 , 3 )+Hgw(3, 5 , 4 )+Hgw(3, 6 , 5 )+Hgw(2, 4 , 4 )+Hgw(2, 5 , 3 )+Hgw(2, 4 , 3 )+Hgw(4, 5 , 4 )+Hgw(2, 4 , 2 )+Hgw(1, 3 , 1 )+Hgw(3, 4 , 2 )+Hgw(4, 4 , 1 )+Hgw(3, 3 , 2 )+Hgw(1, 2 , 1)+Hgw(1, 5 , 1)+Hgw(1, 3 , 2)+Hgw(1, 4 , 2)+Hgw(1, 4 , 3)+Hgw(1, 5 , 2)+Hgw(1, 5 , 3)+Hgw(1, 5 , 4)+Hgw(1, 6 , 2)+Hgw(2, 2 , 1)+Hgw(2, 3 , 1)+Hgw(2, 3 , 2)+Hgw(2, 4 , 1)+Hgw(2, 5 , 1 )+Hgw(2, 6 , 4)+Hgw(2, 6 , 3)+Hgw(2, 6 , 2)+Hgw(2, 6 , 1)+Hgw(3, 3 , 1)+Hgw(3, 4 , 1)+Hgw(3, 5 , 1)+Hgw(3, 5 , 2)+Hgw(3, 5 , 3)+Hgw(3, 6 , 4)+Hgw(3, 6 , 3)+Hgw(3, 6 , 2)+Hgw(4, 5 , 1)+Hgw(4, 5 , 3)+Hgw(4, 6 , 1)+Hgw(4, 6 , 3)+Hgw(4, 6 , 5)+Hgw(5, 5 , 1)+Hgw(5, 5 , 2)+Hgw(5, 6 , 4)+Hgw(5, 6 , 5);
    He(7)=Hgw(1, 1 , 6 )+Hgw(4, 5 , 6 )+Hgw(4, 6 , 6 )+Hgw(2, 5 , 5 )+Hgw(1, 5 , 5 )+Hgw(5, 6 , 6 )+Hgw(1, 6 , 6 )+Hgw(1, 4 , 4 )+Hgw(4, 5 , 5 )+Hgw(2, 4 , 5 )+Hgw(1, 5 , 6 )+Hgw(2, 4 , 6 )+Hgw(3, 4 , 6 )+Hgw(2, 3 , 5 )+Hgw(1, 1 , 5 )+Hgw(1, 1 , 4 )+Hgw(1, 1 , 3 )+Hgw(1, 1 , 2)+Hgw(1, 2 , 2)+Hgw(1, 2 , 3)+Hgw(1, 2 , 4)+Hgw(1, 2 , 5)+Hgw(1, 2 , 6)+Hgw(1, 3 , 3)+Hgw(1, 3 , 4)+Hgw(1, 3 , 5)+Hgw(1, 3 , 6)+Hgw(1, 4 , 5)+Hgw(1, 6 , 5)+Hgw(2, 1 , 5)+Hgw(2, 1 , 6)+Hgw(2, 2 , 5)+Hgw(2, 2 , 6)+Hgw(2, 3 , 4)+Hgw(2, 3 , 6)+Hgw(2, 5 , 4)+Hgw(2, 5 , 6)+Hgw(2, 6 , 6)+Hgw(2, 6 , 5)+Hgw(3, 4 , 5)+Hgw(3, 5 , 5)+Hgw(3, 5 , 6)+Hgw(3, 6 , 6)+Hgw(4, 4 , 6)+Hgw(1, 4 , 6);
    He(8)=Hgw(4, 2 , 2 )+Hgw(6, 5 , 4 )+Hgw(5, 5 , 4 )+Hgw(5, 4 , 4 )+Hgw(6, 4 , 3 )+Hgw(5, 4 , 2 )+Hgw(5, 4 , 1 )+Hgw(5, 3 , 2 )+Hgw(4, 2 , 1 )+Hgw(4, 3 , 2 )+Hgw(3, 2 , 1)+Hgw(3, 2 , 2)+Hgw(4, 3 , 3)+Hgw(5, 3 , 1)+Hgw(5, 4 , 3);
    He(9)=Hgw(6, 6 , 6 )+Hgw(6, 6 , 5 )+Hgw(6, 5 , 5 )+Hgw(6, 5 , 6);
    He(10)=Hgw(4, 4 , 4 )+Hgw(5, 5 , 5 )+Hgw(3, 3 , 3 )+Hgw(3, 4 , 4 )+Hgw(2, 3 , 3 )+Hgw(1, 1 , 1 )+Hgw(2, 2 , 2)+Hgw(3, 3 , 4)+Hgw(3, 4 , 3)+Hgw(4, 4 , 2)+Hgw(4, 4 , 3)+Hgw(4, 4 , 5)+Hgw(5, 5 , 6);
    
    %% Color Properties
    Pcold=sum(sum(sum(Hgw(1:3,:,:))))-Hgw(1,1,1);
    Pdark=sum(sum(sum(Hgw(1:4,1:4,:))))+sum(sum(sum(Hgw(1:2,5:6,1:3))))+sum(sum(sum(Hgw(3:4,5:6,1:2))));
    Ppale=sum(sum(sum(Hgw(4:6,5:6,4:6))));
    Padj=max(max(He(1)+He(2)+He(5),He(3)+He(4)+He(8)),max(He(5)+He(6)+He(7),He(8)+He(9)+He(10)));
    Pcom=max(max(He(1)+He(2)+He(6),He(3)+He(4)+He(5)+He(7)),He(9)+He(10));

end


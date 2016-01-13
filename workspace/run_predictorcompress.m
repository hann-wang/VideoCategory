clear all;
close all;

load('./data/finalobjbw.mat');
load('./data/auxilary.mat');
total=position(5,2);
%calculate scatters
Hscatter=zeros(5,216);
SPscatter=zeros(5,93);
DSPscatter=zeros(5,93);
VDSPscatter=zeros(5,93);
LFPscatter=zeros(5,93);
CPscatter=zeros(5,253);
for t=1:5
    Hcnt=0;
    for i=position(t,1):position(t,2)
        %do not deal with bad videos
        if (data(i,1763)==0 || i==631)
            continue;
        end
        Hscatter(t,:)=Hscatter(t,:)+data(i,1767:1982);
        SPscatter(t,:)=SPscatter(t,:)+data(i,1:93);
        DSPscatter(t,:)=DSPscatter(t,:)+data(i,94:186);
        VDSPscatter(t,:)=VDSPscatter(t,:)+data(i,187:279);
        LFPscatter(t,:)=LFPscatter(t,:)+data(i,280:372);
        CPscatter(t,:)=CPscatter(t,:)+data(i,373:625);
        Hcnt=Hcnt+1;
    end
    Hscatter(t,:)=Hscatter(t,:)/Hcnt;
    SPscatter(t,:)=SPscatter(t,:)/Hcnt;
    DSPscatter(t,:)=DSPscatter(t,:)/Hcnt;
    VDSPscatter(t,:)=VDSPscatter(t,:)/Hcnt;
    LFPscatter(t,:)=LFPscatter(t,:)/Hcnt;
    CPscatter(t,:)=CPscatter(t,:)/Hcnt;
end

%calculate scatter distance and color entropy
for t=1:5
    for i=position(t,1):position(t,2)
        %do not deal with bad videos
        if (data(i,1763)==0)
            continue;
        end
        %global weighted color histogram distance
        for j=1:5
            data(i,1751+j)=sqrt(sum((data(i,1767:1982)-Hscatter(j,:)).^2));
        end
        %color entropy
        color=data(i,1767:1982)/100;
        color(color==0)=[];
        data(i,1751)=-sum(color.*log2(color));
        %CP distance
        for j=1:5
            data(i,671+j)=sqrt(sum((data(i,373:625)-CPscatter(j,:)).^2));
        end
        %LFP distance
        for j=1:5
            data(i,666+j)=sqrt(sum((data(i,280:372)-LFPscatter(j,:)).^2));
        end
        %VDSP distance
        for j=1:5
            data(i,661+j)=sqrt(sum((data(i,187:279)-VDSPscatter(j,:)).^2));
        end
        %DSP distance
        for j=1:5
            data(i,656+j)=sqrt(sum((data(i,94:186)-DSPscatter(j,:)).^2));
        end
        %SP distance
        for j=1:5
            data(i,651+j)=sqrt(sum((data(i,1:93)-SPscatter(j,:)).^2));
        end
    end
end


%Remove zeros
i=1;
cnt=1;
while (cnt<=total)
    if (data(i,1763)==0)
        data(i,:)=[];
    else
        i=i+1;
    end
    cnt=cnt+1;
end

%save rest predictors only
%Hgw
data(:,1767:1982)=[];
%zeros between audio and video
data(:,677:1750)=[];
%SP-CP
data(:,1:625)=[];

%age
age=zeros(1,5);
age(1)=length(find(data(:,end)==1014));
age(2)=length(find(data(:,end)==1016));
age(3)=length(find(data(:,end)==1017));
age(4)=length(find(data(:,end)==1019));
age(5)=length(find(data(:,end)==1020));
save('./data/data4train.mat','data');
save('./data/auxilary.mat','age','position','Hscatter','SPscatter','DSPscatter','VDSPscatter','LFPscatter','CPscatter');

function [ SP,DSP,VDSP,LFP,CP,ED,SCP,ZCR,RMS] = processAudio( dat_aud )
% audio eigenvector extracting
% use six eigenvenctor sp,dsp,vdsp,lfp,scp and cp to represent this
% audio. each eigenvector comes from Short Time FT of the audio. RMS and
% ZCR represent time character. for each audio data, choose 1st column as
% its raw data to solve double-channel problem.

%% dat_aud.rate is 1*1?
%% 20*log10(X)=-inf?



%% Cut Audio
aud=dat_aud.data(:,1);
if (dat_aud.totalDuration>400)
    audiolength=length(aud);
    cutlength=150*dat_aud.rate;
    middle=audiolength/2;
    aud=aud(middle-cutlength:middle+cutlength);
    if(any(aud)==0)
        SP(1,93)=0;
        DSP(1,93)=0;
        VDSP(1,93)=0;
        LFP(1,93)=0;
        CP(1,253)=0;
        SCP(1,23)=0;
        ZCR=0;
        RMS=0;
        ED=0;
        return
    end
    if(dat_aud.rate>29000)
        aud=resample(aud,22000,dat_aud.rate);
        dat_aud.rate=22000;
    end
    ZCR=sum(abs(diff(sign(aud))))/2/(2*cutlength+1)*dat_aud.rate;
    RMS=sqrt(mean(aud.^2));
    [x,f,~,~]=spectrogram(aud,2048,512,2048,dat_aud.rate);
end
if(dat_aud.totalDuration>60&&dat_aud.totalDuration<=400)
        if(any(aud)==0)
            SP(1,93)=0;
            DSP(1,93)=0;
            VDSP(1,93)=0;
            LFP(1,93)=0;
            CP(1,253)=0;
            ED=0;
            SCP(1,23)=0;
            ZCR=0;
            RMS=0;
            return
        end
        if(dat_aud.rate>29000)
            aud=resample(aud,22000,dat_aud.rate);
            dat_aud.rate=22000;
        end
        ZCR=sum(abs(diff(sign(aud))))/2/length(aud)*dat_aud.rate;
        RMS=sqrt(mean(aud.^2));
        [x,f,~,~]=spectrogram(aud,2048,512,2048,dat_aud.rate);
end
if (dat_aud.totalDuration<=60)
        if(any(aud)==0)
            SP(1,93)=0;
            DSP(1,93)=0;
            VDSP(1,93)=0;
            LFP(1,93)=0;
            CP(1,253)=0;
            ED=0;
            SCP(1,23)=0;
            ZCR=0;
            RMS=0;
            return
        end
        if(dat_aud.rate>29000)
            aud=resample(aud,22000,dat_aud.rate);
            dat_aud.rate=22000;
        end
        ZCR=sum(abs(diff(sign(aud))))/2/length(aud)*dat_aud.rate;
        RMS=sqrt(mean(aud.^2));
        [x,f,~,~]=spectrogram(aud,512,256,2048,dat_aud.rate);
end
%% get magnitude
x=abs(x);
[~,b]=size(x);
x=x(:,2:b-1);
m=440*2^(-57/12);
f=1200*log2(f/m);
p=ceil((11650-2050)/100);
F=[2050:100:2050+100*p]';
L=floor((f(end)-2050)/100)+1;
K=zeros(L,1);
for i=1:L
    K(i)=find(f>=F(i),1);
end
X=zeros(L,b-2);
flag=0;
X(1,:)=x(K(1),:);
for i=2:L-1
    if K(i)==K(i+1)
        flag=flag+1;
    else
        X(i,:)=x(K(i),:);
        if(flag>0)
            for j=1:flag
                X(i-j,:)=x(K(i),:)-j*(x(K(i),:)-x(K(i-flag-1),:))/(flag+1);
            end
            flag=0;
        end
    end
end
X(L,:)=sum(x(K(L):end,:),1);
[a,b]=size(X);
%% energy on frequency axis
Energy=mean(X,2);
sumE=zeros(a,1);
sumE(1)=Energy(1);
for i=2:a
    sumE(i)=sumE(i-1)+Energy(i-1);
end
ED=find(sumE>=0.7*sumE(a),1);

%% solve -inf, change 0 to 1E-20
X(X==0)=10^-20;
X=20*log10(X);
S=zeros(a,b);
for j=1:100
    S(:,j)=X(:,j)-mean(X(:,1:j+100),2);
end
for j=101:b-100
    S(:,j)=X(:,j)-mean(X(:,j-100:j+100),2);
end
for j=b-99:b
    S(:,j)=X(:,j)-mean(X(:,j-100:b),2);
end
%% sp
A=zeros(a,floor(b/5)-1);
for j=1:5:5*floor(b/5)-9
    A(:,(j-1)/5+1)=prctile(S(:,j:j+9),90,2);
end
SP=(mean(A,2))';
SP(1,p)=0;
SP=SP(:,1:93);
%% dsp
S_B=abs(S(:,4:end)-S(:,1:b-3));
B=zeros(a,floor((b-3)/5)-4);
for j=1:5:5*floor((b-3)/5)-24
    B(:,(j-1)/5+1)=prctile(S_B(:,j:j+24),90,2);
end
DSP=(mean(B,2))';
DSP(1,p)=0;
DSP=DSP(:,1:93);
%% vdsp
C=zeros(a,floor((b-3)/5)-4);
for j=1:5:5*floor((b-3)/5)-24
    C(:,(j-1)/5+1)=var(S_B(:,j:j+24),0,2);
end
VDSP=(mean(C,2))';
VDSP(1,p)=0;
VDSP=VDSP(:,1:93);
%% lfp
g=linspace(0,dat_aud.rate/512,128+1);
flux=repmat(1./g(2:32)/4+4./g(2:32),a,1);
w=[0.05,0.1,0.25,0.5,1,0.5,0.25,0.1,0.05];
filt1=filter2(w,eye(a));
filt1=filt1./repmat(sum(filt1,2),1,a);
filt2=filter2(w,eye(30));
filt2=(filt2./repmat(sum(filt2,2),1,30))';
d=zeros(floor(b/128)-3,a*30);
for j=1:128:128*floor(b/128)-511
    Y4=fft(X(:,(1:512)+j-1),512,2);
    Y41=abs(Y4(:,2:32)).*flux;
    Y41=filt1*abs(diff(Y41,1,2))*filt2;
    d((j-1)/128+1,:)=Y41(:)';
end
D=zeros(floor(b/128)-3,L);
for j=1:30:30*a-29
    D(:,(j-1)/30+1)=mean(d(:,j:j+29),2);
end
LFP=median(D);
LFP(1,p)=0;
LFP=LFP(:,1:93);
%% scp
S=zeros(floor(L/4),b);
for i=1:floor(L/4)
    S(i,:)=max(X((4*i-3):4*i,:))-min(X((4*i-3):4*i,:));
end
[~,b]=size(S);
F1=zeros(floor(L/4),floor(b/20)-1);
for j=1:20:20*floor(b/20)-39
    Y6=prctile(S(:,j:j+39),10,2);
    F1(:,(j-1)/20+1)=Y6;
end
SCP=(mean(F1,2))';
SCP(1,24)=0;
SCP=SCP(:,1:23);
%% cp
L=floor((f(end)-2050)/400)+1;
p=ceil((11650-2050)/400);
F=[2050:200:2050+400*p]';
K=zeros(L,1);
for i=1:L
    K(i)=find(f>=F(i),1);
end
X=zeros(L,b);
flag=0;
X(1,:)=x(K(1),:);
for i=2:L-1
    if K(i)==K(i+1)
        flag=flag+1;
    else
        X(i,:)=x(K(i),:);
        if(flag>0)
            for j=1:flag
                X(i-j,:)=x(K(i),:)-j*(x(K(i),:)-x(K(i-flag-1),:))/(flag+1);
            end
            flag=0;
        end
    end
end
X(L,:)=sum(x(K(L):end,:),1);
[a,b]=size(X);
X(X==0)=10^-20;
X=20*log10(X);
S=zeros(a,b);
for j=1:100
    S(:,j)=X(:,j)-mean(X(:,1:j+100),2);
end
for j=101:b-100
    S(:,j)=X(:,j)-mean(X(:,j-100:j+100),2);
end
for j=b-99:b
    S(:,j)=X(:,j)-mean(X(:,j-100:b),2);
end
E=zeros(a,a,floor(b/128)-1);
for j=1:128:128*floor(b/128)-255
    Y5=corr(S(:,j:j+255)');
    E(:,:,(j-1)/128+1)=Y5;
end
cp=prctile(E,50,3);
cp(p,p)=0;
CP=zeros(1,300);
for i=2:p-1
    for j=1:i-1
        CP((i*i-3*i+2)/2+j)=cp(i,j);
    end
end
CP=CP(:,1:253);
end


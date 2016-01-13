clear all;
close all;

load('../dataset/TRAINset.mat');
load('./data/final.mat');
[total,~]=size(data);
data(:,end)=TRAINclass;

%Remove zero lines
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

%Remove Zero Columns
data(:,1479:1762)=[];

%Generate data for ANN
[total,~]=size(data);
samples=data(:,1:end-1);
%Single-Spot Encoding Output
result=zeros(total,5);
for i=1:total
    switch(data(i,end))
        case 1014
            result(i,1)=1;
        case 1016
            result(i,2)=1;
        case 1017
            result(i,3)=1;
        case 1019
            result(i,4)=1;
        case 1020
            result(i,5)=1;
    end
end

%% Bagged Tree
Mdl = fitensemble(data(:,1:end-1),data(:,end),'bag',500,'tree','type','classification');

%% Classification Tool shipped with MATLAB R2015a/b
%classificationLearner;

clear all;
close all;


load('../dataset/TRAINset.mat');
total=length(TRAINclass);
data4train=zeros(total,1998);

%Compose video data for confirming model
load('./data/datafilelist.mat');
for i=1:length(datafilelist)
    load(datafilelist{i});
    data4train=data4train+data;
end

%Set standard class
data4train(:,end)=TRAINclass;

%Remove zeros
% i=1;
% cnt=1;
% while (cnt<=total)
    % if (data4train(i,1763)==0)
        % data4train(i,:)=[];
    % else
        % i=i+1;
    % end
    % cnt=cnt+1;
% end

save('./data/data4train.mat','data4train');

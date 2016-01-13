%clear all;
close all;

%% input data

load('../dataset/TESTset.mat');
%N_testdata  =   length(TESTclass);
START_testdata=4;
N_testdata=35;

filenames   =   cell(N_testdata, 1);
groups_std  =   TESTclass(1:N_testdata);
for i=1:N_testdata
    filenames{i} = ['../dataset/train/',num2str(groups_std(i)),'/',TESTfiles{i}];
end

%% process results

groups_rst  =   zeros(N_testdata, 1);
time_used   =   zeros(N_testdata, 1);

prev_rst    =   -1;
global  img_dir;

for i_testdata = START_testdata : N_testdata
    disp(i_testdata);
    filename    =   filenames{i_testdata};
    group_std   =   groups_std(i_testdata);

    % load sample video data and full audio data
    [dat_vid, ~]    =   mmread(filename, [1 : 2], [], false, true);
    [~, dat_aud]    =   mmread(filename, [], [], true, false);

    % convert video to images
    img_dir     =   [filename, '_img/'];
    if (~exist(img_dir, 'dir'))
        mkdir(img_dir);
        mmread(filename, [], [], false, false, 'saveFrame');
    end
    dat_vid.nrFramesTotal   =   numel(dir([img_dir, '*.jpg']));
    dat_vid.filename        =   filename;

    % call function process
    tic;
    group_rst   =   fun_process(dat_vid, dat_aud, img_dir, prev_rst);
    time_used(i_testdata)   =   toc / dat_vid.totalDuration;
    groups_rst(i_testdata)  =   group_rst;
    
    if (group_rst==group_std)
        display('correct');
    else
        display('wrong');
    end

    % update previous group result
    prev_rst    =   group_std;
end

%% performance evaluation
if (START_testdata>1)
    groups_std(1:START_testdata-1)=[];
    groups_rst(1:START_testdata-1)=[];
    time_used(1:START_testdata-1)=[];
end
precision   =   mean(groups_std == groups_rst);
disp(['Processing precision:     ', num2str(precision, '%f')]);
disp(['Relative processing time: ', num2str(mean(time_used), '%f')]);


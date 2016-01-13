clear all;
close all;
clc;

%% setup environment

%addpath('./mexopencv/');

flag_install_mexopencv      =   false;
if (flag_install_mexopencv)
    if (isunix())
        error('Failed to install mexopencv automatically. Please install mexopencv using make.');
    else
        mexopencv.make('opencv_path', 'C:\Users\liuxj\Desktop\opencv\build');
    end
end

flag_compile_mexprocess     =   false;
if (flag_compile_mexprocess)
    if (exist('fun_process.cpp', 'file'))
        mex     -largeArrayDims -I'C:\Users\liuxj\Desktop\opencv\build\include' -L'C:\Users\liuxj\Desktop\opencv\build\x64\vc11\lib'  -lopencv_ts300 -lopencv_world300  './fun_process.cpp' -output './fun_process';
    else
        error('Failed to find file fun_process.cpp.');
    end
end

%% input data

N_testdata  =   2;

filenames   =   cell(N_testdata, 1);
filenames{1}    =   '../dataset/samples/Mode2-WeRMode2388.flv.ogv';
filenames{2}    =   '../dataset/samples/TheOintment-FakeIPhoneCommercial822.flv.ogv';

groups_std  =   zeros(N_testdata, 1);
groups_std(1)   =   1006;
groups_std(2)   =   1020;

%% process results

groups_rst  =   zeros(N_testdata, 1);
time_used   =   zeros(N_testdata, 1);

prev_rst    =   -1;
global  img_dir;
for i_testdata = 1 : N_testdata

    filename    =   filenames{i_testdata};
    group_std   =   groups_std(i_testdata);

    % load sample video data and full audio data
    [dat_vid, ~]    =   mmread(filename, [1 : 10], [], false, true);
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

    % update previous group result
    prev_rst    =   group_std;
end

%% performance evaluation

precision   =   mean(groups_std == groups_rst);
disp(['Processing precision:     ', num2str(precision, '%f')]);
disp(['Relative processing time: ', num2str(mean(time_used), '%f')]);

%% post process
if (flag_compile_mexprocess)
    delete('fun_process.mex*');
end

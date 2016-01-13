function [maxface,faceratio,totalface,blobcnt,blobrate,cornercnt]=processObject(width,height,FramesTotal,img_dir,vidrate,totalDuration,filename,valid)
%Object detection
% We use object detection to determine certain categories
    global ccface detector;
    
    swidth=350;
    sheight=round(height/width*swidth);
    imsize=sheight*swidth;
    
    if (valid)
        seq=1:max(1,floor(FramesTotal/30)):FramesTotal;
    else
        total=floor((floor(totalDuration)-1.5)*vidrate);
        seq=1:max(1,floor(total/30)):total;
        [video,~] = mmread(filename, seq, [], false, true);
        
    end
    
    %% Face detection
    if (isempty(ccface))
    	xmlfile = fullfile('./data/frontalface.xml');
        ccface = cv.CascadeClassifier(xmlfile);
    end
    facearea=zeros(length(seq),1);
    facecnt=zeros(length(seq),1);
    %% Blob detection
    if (isempty(detector))
        detector = cv.FeatureDetector('SimpleBlobDetector');
    end
    blobcnt=0;
    blobrate=0;
    %% Corner detection
    cornercnt=0;
    
    %% Deal with no more than 30 frames
    for t=1:length(seq)
        if (valid)
            im = imread([img_dir, num2str(seq(t), '%06d.jpg')]);
        else
            im = video.frames(t).cdata;
        end
        im = cv.resize(im,[swidth,sheight]);
        boxes = ccface.detect(im);
        facecnt(t)=numel(boxes);
        facesize=zeros(facecnt(t),1);
        for i=1:facecnt(t)
            facesize(i)=boxes{i}(3);
        end
        if (~isempty(boxes))
            maxface=max(facesize);
            facearea(t)=maxface*maxface/imsize;
        end
        im_gray=cv.cvtColor(im, 'RGB2GRAY');
        keypoints = detector.detect(im_gray);
        blobcnt=blobcnt+length(keypoints);
        for i=1:length(keypoints)
            blobrate=blobrate+sum(keypoints(i).size)/imsize;
        end
        dst0=cv.cornerHarris(im_gray);
        dst1=cv.normalize(dst0,'NormType','MinMax');
        p=cv.convertScaleAbs(dst1);
        cornercnt=cornercnt+length(find(p~=0))/imsize;
    end
    facearea(facearea==0)=[];
    facecnt(facecnt==0)=[];
    maxface=max(facearea);
    if (isempty(maxface) || maxface==0)
        maxface=0;
        faceratio=0;
        totalface=0;
    else
        average=mean(facearea);
        faceratio=average/maxface;
        totalface=mean(facecnt);
    end
    blobrate=blobrate/length(seq);
    blobcnt=blobrate/length(seq);
    cornercnt=cornercnt/length(seq);
end


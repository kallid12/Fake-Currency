 
clc;
clear all;
close all;
%cd DATASETS\

df=[]

for i = 1:20
    i
    B=imread(strcat(int2str(i),'.jpg'));
    I=imresize(B,0.5);

   %Image Preprocessing
  [BW,maskedRGBImage] = createMask(I);
            seg_img = maskedRGBImage;

            B = rgb2gray(seg_img);
            SE = strel('rectangle',[40 30]);
            img=imopen(B,SE);

            glcms = graycomatrix(img);

            stats = graycoprops(glcms,'Contrast Correlation Energy Homogeneity');

            Contrast = stats.Contrast;
            Energy = stats.Energy;
            Homogeneity = stats.Homogeneity;
            Mean = mean2(seg_img);
            Standard_Deviation = std2(seg_img);
            Entropy = entropy(seg_img);
            RMS = mean2(rms(seg_img));
            %Skewness = skewness(img)
            Variance = mean2(var(double(seg_img)));
            a = sum(double(seg_img(:)));
            Smoothness = 1-(1/(1+a));
            % Inverse Difference Movement
            m = size(seg_img,1);
            n = size(seg_img,2);
            in_diff = 0;
            for i = 1:m
                for j = 1:n
                    temp = seg_img(i,j)./(1+(i-j).^2);
                    in_diff = in_diff+temp;
                end
            end
            IDM = double(in_diff);

            Fr = horzcat(1,[Contrast,Energy,Homogeneity, Mean, Standard_Deviation, Entropy, RMS, Variance, Smoothness, IDM]);


   df=[df;Fr];


end
cd ..


%% Image Acquisition
            [fname,path]=uigetfile('.jpg','Test için görüntü seçin');
filename=strcat(path,fname);
B=imread(filename);
imshow(B),figure
I=imresize(B,0.5);

%% Image Preprocessing
            imshow(I);
            figure 
            title ('Original Image');
          [BW,maskedRGBImage] = createMask(I); %Image Segmentation
            seg_img = maskedRGBImage;

            imshow(BW);
            figure
imshow(maskedRGBImage)

            B = rgb2gray(seg_img);
            SE = strel('rectangle',[40 30]);
            img=imopen(B,SE);
            imshow(img)


            glcms = graycomatrix(img);

            stats = graycoprops(glcms,'Contrast Correlation Energy Homogeneity');

            Contrast = stats.Contrast;
            Energy = stats.Energy;
            Homogeneity = stats.Homogeneity;
            Mean = mean2(seg_img);
            Standard_Deviation = std2(seg_img);
            Entropy = entropy(seg_img);
            RMS = mean2(rms(seg_img));
            
            Variance = mean2(var(double(seg_img)));
            a = sum(double(seg_img(:)));
            Smoothness = 1-(1/(1+a));
           
            m = size(seg_img,1);
            n = size(seg_img,2);
            in_diff = 0;
            for i = 1:m
                for j = 1:n
                    temp = seg_img(i,j)./(1+(i-j).^2);
                    in_diff = in_diff+temp;
                end
            end
            IDM = double(in_diff);

            Testftr = horzcat(1,[Contrast,Energy,Homogeneity, Mean, Standard_Deviation, Entropy, RMS, Variance, Smoothness, IDM]);


   % SVM eğitim
   TrainingSet=df;
   GroupTrain={'1','1','1','1','1','1','1','1','1','1','2','2','2','2','2','2','2','2','2','2'};
   TestSet=Testftr;

   %SVM
   Y=GroupTrain;
   classes=unique(Y);
   SVMModels=cell(length(classes),1);
   rng(1);   

   for j=1:numel(classes)
       idx=strcmp(Y',classes(j));
       SVMModels{j}=fitcsvm(df,idx,'ClassNames',[false true],'Standardize',true,'KernelFunction','rbf','BoxConstraint',1)
   end
   xGrid=Testftr;
   for j=1:numel(classes)
   [~,score]=predict(SVMModels{j},xGrid)
   Scores(:,j)=score(:,2);
   end


   [~,maxScore]=max(Scores,[],2)

   result=maxScore;

   if result == 1
       msgbox('Gerçek')
   elseif result == 2
      msgbox('Sahte');
   
   end


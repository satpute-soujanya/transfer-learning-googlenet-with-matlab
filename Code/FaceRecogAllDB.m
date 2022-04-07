clc;
clear all;
close all;
%pool(parpool)
cd_in=cd;
imds = imageDatastore('FinalDataset', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames'); 
[imdsTrain,imdsValidation,imdsTest] = splitEachLabel(imds,0.7,0.1);
 augimdsTrain = augmentedImageDatastore([224,224],imdsTrain,"ColorPreprocessing","gray2rgb");
 augimdsValidation = augmentedImageDatastore([224,224],imdsTrain,"ColorPreprocessing","gray2rgb");
augimdsTest = augmentedImageDatastore([224,224],imdsTest,"ColorPreprocessing","gray2rgb");
net = googlenet;
lgraph = layerGraph(net);
[learnableLayer,classLayer] = findLayersToReplace(lgraph);
numClasses = numel(categories(imdsTrain.Labels));
if isa(learnableLayer,'nnet.cnn.layer.FullyConnectedLayer')
    newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
    
elseif isa(learnableLayer,'nnet.cnn.layer.Convolution2DLayer')
    newLearnableLayer = convolution2dLayer(1,numClasses, ...
        'Name','new_conv', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
end

lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);
newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);
miniBatchSize = 64;
valFrequency = floor(numel(augimdsTrain.Files)/miniBatchSize);
options = trainingOptions('sgdm', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',50, ...
    'InitialLearnRate',3e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',valFrequency, ...
    'Plots','training-progress');
net = trainNetwork(augimdsTrain,lgraph,options)
[YPred,scores] = classify(net,augimdsTest);
disp(['Accuracy = ',num2str(100*sum(imdsTest.Labels==YPred)/size(YPred,1))]);
%IZendaya = imread('Zendaya.jpg');
%ITaylorSwift = imread('Taylor Swift.jpg');
%IjeffBezos = imread('Jeff Bezos.jpg');
%IZendaya = imresize(IZendaya,[224,224]);
%ITaylorSwift = imresize(ITaylorSwift,[224,224]);
%IjeffBezos = imresize(IjeffBezos,[224,224]);
%[ZLable, ZScore]= classify(net,IZendaya)
%[TSLabel, TSScore]= classify(net,ITaylorSwift)
%[JBLabel, JBScore]= classify(net,IjeffBezos)





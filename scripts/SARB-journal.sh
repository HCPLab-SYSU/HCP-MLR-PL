#!/bin/bash

# Make for COCOAPI
# cd cocoapi/PythonAPI
# make -j8
# cd ../..

post='SARB-journal-COCO'
printFreq=1000

mode='SARB-journal'
dataset='COCO2014'
prob=0.5

pretrainedModel='./data/checkpoint/resnet101.pth'
resumeModel='None' 
evaluate='False'

epochs=20
startEpoch=0
stepEpoch=10

batchSize=8
lr=1e-5
momentum=0.9
weightDecay=5e-4

cropSize=448
scaleSize=512
workers=8

mixupEpoch=5
contrastiveLossWeight=0.05

topK=1
recomputePrototypeInterval=5

isAlphaLearnable='True'
isBetaLearnable='True'

# use single gpu (eg,gpu 0) to trian:
#     CUDA_VISIBLE_DEVICES=0 
# use multiple gpu (eg,gpu 0 and 1) to train
#     CUDA_VISIBLE_DEVICES=0,1  
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python SARB-journal.py \
    --post ${post} \
    --printFreq ${printFreq} \
    --mode ${mode} \ 
    --dataset ${dataset} \
    --prob ${prob}\
    --pretrainedModel ${pretrainedModel} \
    --resumeModel ${resumeModel} \
    --evaluate ${evaluate} \
    --epochs ${epochs} \
    --startEpoch ${startEpoch} \
    --stepEpoch ${stepEpoch} \
    --batchSize ${batchSize} \
    --lr ${lr} \
    --momentum ${momentum} \
    --weightDecay ${weightDecay} \
    --cropSize ${cropSize} \
    --scaleSize ${scaleSize} \
    --workers ${workers} \
    --mixupEpoch ${mixupEpoch} \
    --contrastiveLossWeight ${contrastiveLossWeight} \
    --topK ${topK} \
    --recomputePrototypeInterval ${recomputeInterval} \
    --isAlphaLearnable ${isAlphaLearnable} \
    --isBetaLearnable ${isBetaLearnable} \
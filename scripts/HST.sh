#!/bin/bash

# Make for COCOAPI
# cd cocoapi/PythonAPI
# make -j8
# cd ../..

post='HST-COCO'
printFreq=1000

mode='HST'
dataset='COCO2014'
prob=0.5

pretrainedModel='./data/checkpoint/resnet101.pth'
resumeModel='None'
evaluate='False'

epochs=20
startEpoch=0
stepEpoch=10

batchSize=32
lr=1e-5
momentum=0.9
weightDecay=5e-4

cropSize=448
scaleSize=512 
workers=8

generateLabelEpoch=5

intraMargin=1.0
isIntraMarginLearnable='True'

intraBCEWeight=0.1
intraCooccurrenceWeight=10.0

interMargin=1.0
isInterMarginLearnable='True'

interBCEWeight=0.1
interInstanceDistanceWeight=0.05
interPrototypeDistanceWeight=0.05

prototypeNumber=10
useRecomputePrototype='True'
computePrototypeEpoch=5

OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=2 python HST.py \
    --post ${post} \
    --printFreq ${printFreq} \
    --mode ${mode} \
    --dataset ${dataset} \
    --prob ${prob} \
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
    --generateLabelEpoch ${generateLabelEpoch} \
    --intraMargin ${intraMargin} \
    --isIntraMarginLearnable ${isIntraMarginLearnable} \
    --intraBCEWeight ${intraBCEWeight} \
    --intraCooccurrenceWeight ${intraCooccurrenceWeight} \
    --interMargin ${interMargin} \
    --isInterMarginLearnable ${isInterMarginLearnable} \
    --interBCEWeight ${interBCEWeight} \
    --interInstanceDistanceWeight ${interInstanceDistanceWeight} \
    --interPrototypeDistanceWeight ${interPrototypeDistanceWeight} \
    --prototypeNumber ${prototypeNumber} \
    --useRecomputePrototype ${useRecomputePrototype} \
    --computePrototypeEpoch ${computePrototypeEpoch} \

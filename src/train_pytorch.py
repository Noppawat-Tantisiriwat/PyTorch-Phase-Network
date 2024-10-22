import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


import numpy as np
import pandas as pd
import csv

import seaborn as sns
import matplotlib.pyplot as plt

import os
import sys
from pathlib import Path
from glob import glob
import time


import model as model
import pipeline as pipeline
import util as util


# ------------------ PIPELINE ------------------

DATA_DIR = "../data/Training_data/pkl/walking_left/LeftFoot"
DATA_ROOT = "../data"

DEVICE = "cpu"
torch.set_default_device(DEVICE)


EndSessionPlusOne = []
with open(os.path.join(DATA_ROOT, 'sessSameRatEnd_plusone_1.csv')) as csvfile2:
	readCSV2 = csv.reader(csvfile2, delimiter=',')
	EndRatSessionPlusOne = []
	end_session_dict = {}
	for i, row2 in enumerate(readCSV2):
		sessSameRatEnd_plusone = row2[0]
		EndRatSessionPlusOne.append(int(sessSameRatEnd_plusone))
		end_session_dict[i] = int(sessSameRatEnd_plusone)


# Getting Dataset
dataset = pipeline.GaitPhasingDataset(
    data_root=DATA_ROOT,
    data_dir=DATA_DIR,
    img_dir=DATA_DIR,
    end_session_plus_one_path="../data/EndSessionPlusOne.csv",
    windowWing=3,
    regressWing=3,
    predictionGap=1,
    rotationMatrixList=None,
    meanForPhaseExtraction=None,
    sdForPhaseExtraction=None,
)


# Training Setting

gpuID=-1
crossValidateNo=-1
system="walking_left"
exercise="walking"
SubjectAmount = dataset.SubjectAmount
startNewTraining = True
sessionFile="../data/test_new_pipe/" + f"{dataset.trainName}phaseModel.ckpt"
imageFolder = "../data/test_new_pipe/img/"
imageFolderRat = "../data/test_new_pipe/img/rat/"


## dataset 
nowState = dataset.AllData["XY"]
nextState = dataset.AllData["XY2"]
sessionSegmentMatrix = dataset.AllData["SessionSegmentMatrix"]
centerPose = dataset.AllData["centerPose"]
srcColorList = dataset.AllData["srcColorList"]
srcColorList_rat = dataset.srcColorList_subject

nowState_img = dataset.AllData_img["XY"]
nextState_img = dataset.AllData_img["XY2"]
sessionSegmentMatrix_img = dataset.AllData_img["SessionSegmentMatrix"]
centerPose_img = dataset.AllData_img["centerPose"]
srcColorList_img = dataset.AllData_img["srcColorList"]


EndNumberPlusOne = dataset.EndNumberPlusOne
EndSessionPlusOne = dataset.EndSessionPlusOne


# Training Args
repeat = False


# pn=PhaseNetwork(2, dataset.windowWing, gpuID)


startTime=time.time()


to_tensor = model.ToTensor(dtype=torch.float, device=DEVICE)
add_noise = model.AddNoise(mean=0, stddev=0.5)


# ------------------ TRAINING ------------------


# Getting model and loss function dict

pn = model.PhaseNetwork(
    h=25,
    depth=4,
    D=2,
    windowWing=3,
    activation="tanh",
    device=DEVICE
)

# Getting model optimizer

optimizer = optim.Adam(pn.parameters())

# dict of loss functions and weights

Ap = -2.3*np.pi/180
Bp = 4*np.pi/180
Cp = (-180 + 45)*np.pi/180


loss_fn_dict = {
    "SpeedPenalty" : {
        "loss" : model.SpeedLoss(
            Ap=Ap,
            Bp=Bp,
            Cp=Cp,
        ),
        "weight": 1.
    },
    "DistributionPenalty" : {
        "loss" : model.DistributionLoss(kind="bad"),
        "weight" : 0.45
    },
    "SingularityPenalty" : {
        "loss" : model.SingularityLoss(),
        "weight" : 0.55
    },
    "MarginPenalty" : {
        "loss" : model.MarginalLoss(),
        "weight" : 0.55
    }
}


EPOCH = 10_000
centerSigma = 1
sessionSegmentMatrix = to_tensor(sessionSegmentMatrix).to(DEVICE)
pn = pn.float().to(DEVICE)
startTime = time.time()

with torch.no_grad():
    
    nowState_tensor = to_tensor(nowState)
    nextState_tensor = to_tensor(nextState)


    # Adding noise to the curent input
    noisyInput = add_noise(nowState_tensor).to(DEVICE)
    nowState_tensor = nowState_tensor.to(DEVICE)
    nextState_tensor = nextState_tensor.to(DEVICE)

    # Compute phase of noisyInput
    prePhase3, phaseXY3, phaseRad3 = pn(noisyInput)

    # Define phase margin for vizualization
    prePhaseMargin = prePhase3

    # Comput phase of nowState
    prePhase, phaseXY, phaseRad = pn(nowState_tensor)

    # Comput phase of nextState
    prePhase2, phaseXY2, phaseRad2 = pn(nextState_tensor)


    speed_penalty = loss_fn_dict["SpeedPenalty"]["weight"] * loss_fn_dict["SpeedPenalty"]["loss"](phaseXY, phaseXY2)
    singularity_penalty = loss_fn_dict["SingularityPenalty"]["weight"] * loss_fn_dict["SingularityPenalty"]["loss"](centerSigma, prePhase) 
    distribution_penalty = loss_fn_dict["DistributionPenalty"]["weight"] * loss_fn_dict["DistributionPenalty"]["loss"](phaseXY, sessionSegmentMatrix)
    margin_penalty = loss_fn_dict["MarginPenalty"]["weight"] * loss_fn_dict["MarginPenalty"]["loss"](centerSigma, prePhaseMargin)

    loss = speed_penalty + singularity_penalty + distribution_penalty + margin_penalty


print("start cost", f"[{loss.item()}, {speed_penalty.item()}, {distribution_penalty.item()},{singularity_penalty.item()},  {margin_penalty.item()}]")

fig0=plt.figure(figsize=(8, 8))#reuse fig it will save memory
fig=[]
for ifig in range(0,SubjectAmount):
    fig.append(plt.figure(figsize=(8, 8)))
figRatF1=[]
for ifigRatF1 in range(0,SubjectAmount):
    figRatF1.append(plt.figure(figsize=(8, 8)))
figRatF2=[]
for ifigRatF2 in range(0,SubjectAmount):
    figRatF2.append(plt.figure(figsize=(8, 8)))

for epoch in range(EPOCH + 1):

    step = epoch

    # Clear gradient
    optimizer.zero_grad()


    # Convert To Tensor
    nowState_tensor = to_tensor(nowState)
    nextState_tensor = to_tensor(nextState)


    # Adding noise to the curent input
    noisyInput = add_noise(nowState_tensor).to(DEVICE)
    nowState_tensor = nowState_tensor.to(DEVICE)
    nextState_tensor = nextState_tensor.to(DEVICE)
    
    # Compute phase of noisyInput
    prePhase3, phaseXY3, phaseRad3 = pn(noisyInput)

    # Define phase margin for vizualization
    prePhaseMargin = prePhase3

    # Comput phase of nowState
    prePhase, phaseXY, phaseRad = pn(nowState_tensor)

    # Comput phase of nextState
    prePhase2, phaseXY2, phaseRad2 = pn(nextState_tensor)

    # Calculate losses

    # print(loss_fn_dict["SpeedPenalty"]["loss"](phaseXY, phaseXY2))

    speed_penalty = loss_fn_dict["SpeedPenalty"]["weight"] * loss_fn_dict["SpeedPenalty"]["loss"](phaseXY, phaseXY2)
    singularity_penalty = loss_fn_dict["SingularityPenalty"]["weight"] * loss_fn_dict["SingularityPenalty"]["loss"](centerSigma, prePhase) 
    distribution_penalty = loss_fn_dict["DistributionPenalty"]["weight"] * loss_fn_dict["DistributionPenalty"]["loss"](phaseXY, sessionSegmentMatrix)
    margin_penalty = loss_fn_dict["MarginPenalty"]["weight"] * loss_fn_dict["MarginPenalty"]["loss"](centerSigma, prePhaseMargin)


    loss = speed_penalty + singularity_penalty + distribution_penalty + margin_penalty

    # back progpagation

    loss.backward()

    # update weight

    optimizer.step()

    # if epoch%50 == 0:

        # print(f"{epoch}: [{}, {}, {}, {}]")

    # print(epoch + 1)
    if epoch % 50 == 0:
        print(f"{epoch} [{loss.item()}, {speed_penalty.item()}, {distribution_penalty.item()},{singularity_penalty.item()},  {margin_penalty.item()}]")
    
    if epoch % 500 == 0:
        torch.save(pn.state_dict(), sessionFile)

    if (epoch<20000 and epoch%1000==0) or (epoch<2000 and epoch%250==0) or (epoch<250 and epoch%50==0) or epoch%2000==0:
        
        with torch.no_grad():

            prePhase_distribution, phaseXY_distribution, phaseRad=pn(nowState_tensor)
            prePhaseMargin, _, _ = pn(add_noise(nowState_tensor).to(DEVICE))

            prePhase_distribution = prePhase_distribution.detach().cpu().numpy().T
            phaseXY_distribution = phaseXY_distribution.detach().cpu().numpy().T
            phaseRad = phaseRad.detach().cpu().numpy().T
            prePhaseMargin = prePhaseMargin.detach().cpu().numpy().T

            #util.savePrePhaseAndPhasePlot(prePhase_distribution,phaseXY_distribution,imageFolder+str(step)+'.png',fig, srcColorList)
            util.saveMarginPlot(prePhase_distribution,phaseXY_distribution,prePhaseMargin,imageFolder+"pytorch_"+str(step)+'.png',fig0, srcColorList)

            util.saveMarginPlot(prePhase_distribution[:,0:EndNumberPlusOne[0]],phaseXY_distribution[:,0:EndNumberPlusOne[0]],prePhaseMargin[:,0:EndNumberPlusOne[0]],imageFolderRat+"pytorch_"+'Rat1_prephase_'+str(step)+'.png',fig[0], srcColorList_rat[0])

            for ipic in range(1,SubjectAmount):
                util.saveMarginPlot(prePhase_distribution[:,EndNumberPlusOne[ipic-1]:EndNumberPlusOne[ipic]],phaseXY_distribution[:,EndNumberPlusOne[ipic-1]:EndNumberPlusOne[ipic]],prePhaseMargin[:,EndNumberPlusOne[ipic-1]:EndNumberPlusOne[ipic]],imageFolderRat+"pytorch_"+'Rat'+str(ipic+1)+'_prephase_'+str(step)+'.png',fig[ipic], srcColorList_rat[ipic])

            #util.saveScatterPlot(phaseXY_distribution,imageFolder+str(step)+'.png',fig)

            #LP
            # phaseRad,prePhase=pn.sess.run([pn.phaseRad,pn.prePhase],feed_dict={pn.inputPlace:nowState})
            phase=phaseRad%(2*np.pi)	#this is cleaner 
            #print("phase:"+str(phase[1:4755]))

            phaseRat=[]
            phaseRat.append(phaseRad[0:EndNumberPlusOne[0]]%(2*np.pi))
            for iphaseRat in range(1,SubjectAmount):
                phaseRat.append(phaseRad[EndNumberPlusOne[iphaseRat-1]:EndNumberPlusOne[iphaseRat]]%(2*np.pi))

            util.saveRatPhaseRainbowPlot(centerPose_img[:,0:EndNumberPlusOne[0]],phaseRat[0], imageFolderRat+"pytorch_"+'Trajec_Rat1_F1_'+str(step)+'.png', figRatF1[0])
            # 				util.saveRatPhaseRainbowPlot2(centerPose_img[:,0:RatEndNumberPlusOne[0]],phaseRat[0], imageFolderRat+'Trajec_Rat1_F2_'+str(step)+'.png', figRatF2[0])
            for iTrajec in range(1,SubjectAmount):
                util.saveRatPhaseRainbowPlot(centerPose_img[:,EndNumberPlusOne[iTrajec-1]:EndNumberPlusOne[iTrajec]],phaseRat[iTrajec], imageFolderRat+"pytorch_"+'Trajec_Rat'+str(iTrajec+1)+'_F1_'+str(step)+'.png', figRatF1[iTrajec])
            # 					util.saveRatPhaseRainbowPlot2(centerPose_img[:,RatEndNumberPlusOne[iTrajec-1]:RatEndNumberPlusOne[iTrajec]],phaseRat[iTrajec], imageFolderRat+'Trajec_Rat'+str(iTrajec+1)+'_F2_'+str(step)+'.png', figRatF2[iTrajec])
                    #util.saveRatPhaseRainbowPlot3(centerPose_img[:,RatEndNumberPlusOne[iTrajec-1]:RatEndNumberPlusOne[iTrajec]],phaseRat[iTrajec], imageFolderRat+'Trajec_Rat'+str(iTrajec+1)+'_H1_'+str(step)+'.png', figRatH1[iTrajec])
                    #util.saveRatPhaseRainbowPlot4(centerPose_img[:,RatEndNumberPlusOne[iTrajec-1]:RatEndNumberPlusOne[iTrajec]],phaseRat[iTrajec], imageFolderRat+'Trajec_Rat'+str(iTrajec+1)+'_H2_'+str(step)+'.png', figRatH2[iTrajec])

            util.savePhaseRainbowPlot(centerPose_img,phase, imageFolder+"pytorch_"+'RatAll_F1_'+str(step)+'.png', fig=None)
            # 				util.savePhaseRainbowPlot2(centerPose_img,phase, imageFolder+'RatAll_F2_'+str(step)+'.png', fig=None)

    if epoch%2000==0:
        print("CV:"+str(crossValidateNo))
        print("Time(min):",(time.time()-startTime)/60)
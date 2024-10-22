### PyTorch implementation version of the model from 978-1-5386-3646-6/18/$31.00 ©2018 IEEE
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import pandas as pd

DEVICE = "cpu"

class ToTensor(nn.Module):

    def __init__(self, dtype=float, device=DEVICE):

        super().__init__()

        self.dtype = dtype
        self.device = device
    
    def forward(self, x):

        return torch.tensor(x, dtype=self.dtype, device=self.device)
    

class AddNoise(nn.Module):

    def __init__(self, mean=0, stddev=1):

        super().__init__()

        self.mean = mean
        self.stddev = stddev

    def forward(self, x):
        
        inputPose = x[0, :, :, :]
        inputHead = x[1, :, :, :]

        randomShape = inputPose.shape
        

        noisyPose = inputPose + torch.normal(mean=self.mean, std=self.stddev, size=randomShape, dtype=torch.float32, device=DEVICE)
        noisyHead = F.normalize(inputHead + torch.normal(mean=self.mean, std=self.stddev, size=randomShape, dtype=torch.float32, device=DEVICE), dim=1)
        x = torch.stack([noisyPose, noisyHead], dim=0)

        return x


    
class PhaseNetwork(nn.Module):

    def __init__(
        self,
        h=25,
        depth=4,
        D=2,
        windowWing=3,
        activation="tanh",
        device=DEVICE
    ):
        super().__init__()

        windowSize = 1 + windowWing * 2 
        self.inputSize =  D * 2 * windowSize
        self.outputSize = 2

        self.hidden_sizes = [self.inputSize] + [h]*depth + [self.outputSize]

        self.h = h
        self.depth = depth
        self.D = D
        self.windowWing = windowWing
        self.deice = device
        

        self.activation = F.tanh if activation=="tanh" else F.relu

        self.layers = self._build_layer()
    

    def _build_layer(self):

        self.fcs = nn.ModuleDict()

        for i in range(len(self.hidden_sizes) - 1):

            nn_input = self.hidden_sizes[i]
            nn_output = self.hidden_sizes[i + 1]

            linear = nn.Linear(
                in_features=nn_input,
                out_features=nn_output,
            )

            ## Weight initialization
            nn.init.normal_(linear.weight, mean=0, std=np.sqrt(1./ nn_input)) # Normal initiation for Weights
            nn.init.zeros_(linear.bias) # Zeros initiation for bias

            self.fcs[f"LinearLayer{i + 1}"] = linear

    def forward(self, x):

        # flatInput = x.reshape(-1, self.inputSize)
        flatInput = x.reshape(28, -1).T

        A = [flatInput]

        for i, (layer_name, layer) in enumerate(self.fcs.items(), 1):
            if i == len(self.fcs):
                # Bypassing activation in the last layer
                Ai = layer(A[i - 1])
            
            else:
                Ai = self.activation(layer(A[i - 1]))

            A.append(Ai)

        prePhase = A[-1]
        # print(prePhase.shape)
        ## calculate phaseXY
        phaseXY = F.normalize(prePhase, p=2, dim=1) # L2 normalization of the prePhase
        # print(phaseXY.shape)
        phaseRad = torch.atan2(phaseXY[:, 1], phaseXY[:, 0]) # Get the phase angle in Radian

        return prePhase, phaseXY, phaseRad
    


# Loss Functions

# Speed Penalty

class SpeedLoss(nn.Module):

    def __init__(self, Ap, Bp, Cp):

        super().__init__()

        self.Ap = Ap
        self.Bp = Bp
        self.Cp = Cp

    def forward(self, currentPhaseXY, nextPhaseXY):

        # calculalte the progress dicection of phase difference

        crossProductZ = torch.mul(currentPhaseXY[:, 0], nextPhaseXY[:, 1]) - torch.mul(currentPhaseXY[:, 1], nextPhaseXY[:, 0]) # poseitive = counter-clockwise
        dotProduct = torch.sum(torch.mul(currentPhaseXY, nextPhaseXY), dim=1)
        phaseProgress = torch.atan2(crossProductZ, dotProduct)

        # Compute penalty of progress within [Cp, Ap)
        isInCA = torch.logical_and(torch.greater_equal(phaseProgress, self.Cp), torch.less(phaseProgress, self.Ap))
        
        y1 = 0
        y2 = 0.5 * np.pi
        x1 = self.Cp
        x2 = self.Ap

        CA_penalty = torch.cos((phaseProgress * (y2 - y1) + y1 * x2 - y2 * x1) / (x2 - x1))
        # print(CA_penalty)
        # Compute penalty of pregress within [Ap, Bp]
        isInAB = torch.logical_and(torch.greater_equal(phaseProgress, self.Ap), torch.less_equal(phaseProgress, self.Bp))
        AB_penalty = torch.empty(phaseProgress.shape).fill_(0.0)
        # print(AB_penalty)
        # Compute penalty of progress >= Bp or <= Cp
        y1 = -0.5 * np.pi
        y2 = 0
        x1 = self.Bp
        x2 = 2 * np.pi + self.Cp

        aboveB_penalty = torch.cos((phaseProgress * (y2 - y1) + y1 * x2 - y2 * x1) / (x2 - x1))
        belowC_penalty = torch.cos(((phaseProgress + 2 * np.pi) * (y2 - y1) + y1 * x2 - y2 * x1) / (x2 - x1)) ## below C is shifted to be above pi

        isAboveB = torch.greater(phaseProgress, self.Bp)
        # print(torch.greater(phaseProgress, self.Bp))
        speedPenalty = torch.mean(torch.where(isInCA,CA_penalty,torch.where(isInAB,AB_penalty,torch.where(isAboveB, aboveB_penalty, belowC_penalty))))

        return speedPenalty
    

class DistributionLoss(nn.Module):

    def __init__(self, kind="bad"):

        super().__init__()

        self.kind = kind

    def forward(self, phaseXY, sessionSegmentMatrix):

        avgCentroidOfEachSession = torch.matmul(phaseXY.T, sessionSegmentMatrix) / torch.sum(sessionSegmentMatrix, dim=0, keepdim=True)
        radiusSquareOfEachSession = torch.sum(torch.square(avgCentroidOfEachSession), dim=1)

        if self.kind == "max":
            maxDistributionPenalty = torch.max(radiusSquareOfEachSession)

            return maxDistributionPenalty

        else:
            badDistributionPenalty = torch.sum(torch.square(torch.mean(phaseXY, axis=0)))

            return badDistributionPenalty
        

class SingularityLoss(nn.Module):
    
    def forward(self, centerSigma, prePhase):

        return torch.mean((1 / (centerSigma * np.sqrt(2 * np.pi))) * torch.exp(-0.5 * torch.square(torch.norm(prePhase, dim=1) / centerSigma)))
    

class MarginalLoss(nn.Module):

    def forward(self, centerSigma, prePhaseMargin):

        return torch.mean((1 / (centerSigma * np.sqrt(2 * np.pi))) * torch.exp(-0.5 * torch.square(torch.norm(prePhaseMargin, dim=1) / centerSigma)))


if __name__ == "__main__":

    ## test loss and training

    pass
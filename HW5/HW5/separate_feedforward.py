from __future__ import print_function, division, absolute_import, unicode_literals
import six
import argparse
import os
import numpy as np
import Model
import torch
import util
import sys

def tensorTransform(Tensor,colStack):
    """
    Transposes the Matrix and selects the given
    number of column.
    Returns a list of tensors for iteration
    """
    TensorT = torch.transpose(Tensor,0,1)
    i =0
    TensorTlist = []
    while i<TensorT.shape[0]:
        TensorT_col = TensorT[i:i+colStack,:]
        TensorTlist.append(TensorT_col)
        i+=colStack

    return TensorTlist

if __name__=="__main__":
    blockSize = 4096
    hopSize = 2048

    if len(sys.argv) != 3:
        print("Usage:\n", sys.argv[0], "input_path output_path")
        exit(1) 

    #read the wav file
    x, fs = util.wavread(sys.argv[1])
    #downmix to single channel
    x = np.mean(x,axis=-1)
    #perform stft
    S = util.stft_real(x, blockSize=  blockSize,hopSize=hopSize)
    magnitude = np.abs(S).astype(np.float32)
    angle = np.angle(S).astype(np.float32)

    #initialize the model
    model = Model.ModelSingleStep(blockSize)

    #load the pretrained model
    checkpoint = torch.load("savedModel_feedforward_best.pt", map_location=lambda storage, loc:storage)
    model.load_state_dict(checkpoint['state_dict'])

    #switch to eval mode
    model.eval()
    with torch.no_grad():
        ###################################
        #Run your Model here to obtain a mask
        ###################################
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        ex = []
        mixture = torch.from_numpy(S.real).to(device)
        mixedTlist = tensorTransform(mixture, 5)
        
        T = len(mixedTlist)
        mask = [] 
        loss_T = 0
        predTensor = []
        model = model.float()
        for tau in range(T):
            out = model.forward(mixedTlist[tau].float())
            mask.append(out)
            
            pred_v = out*mixedTlist[tau].float()
            predTensor.append(pred_v)
            #loss = KL_loss(pred_v,targetT[tau])
       
        combineTensor = torch.cat(predTensor)
        magnitude_masked = combineTensor.transpose(0,1)
        
        magnitude_masked = magnitude_masked.numpy()

    ###################################


    #perform reconstruction
    y = util.istft_real(magnitude_masked * np.exp(1j * angle), blockSize=blockSize, hopSize=hopSize)

    #save the result
    util.wavwrite(sys.argv[2], y,fs)


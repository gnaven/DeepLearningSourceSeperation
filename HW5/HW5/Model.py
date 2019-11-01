from __future__ import print_function, division, absolute_import, unicode_literals
import six
import os
import numpy as np
import Data
import torch
import torchvision.transforms as transforms 
import torch.nn as nn
import argparse
import torch.nn.functional as F

import matplotlib.pyplot as plt
from tqdm import tqdm, trange



class ModelSingleStep(torch.nn.Module):
    def __init__(self, blockSize):
        super(ModelSingleStep, self).__init__()
        self.blockSize = blockSize

        ###################################
        #define your layers here
        ###################################
        self.fc1 = nn.Linear(2049,1000)
        self.fc2 = nn.Linear(1000,400)

        self.fc3 = nn.Linear(400,1000)
        self.fc4 = nn.Linear(1000,2049)

        ###################################

        self.initParams()

    def initParams(self):
        for param in self.parameters():
            if len(param.shape)>1:
                torch.nn.init.xavier_normal_(param)


    def encode(self, x):
        ###################################
        #implement the encoder
        ###################################
        #x = x.view(x.shape[0],-1)

        x = F.leaky_relu(self.fc1(x))
        h = F.leaky_relu(self.fc2(x))

        ###################################
        return h

    def decode(self, h):
        ###################################
        #implement the decoder
        ###################################
        h = F.leaky_relu(self.fc3(h))
        o = F.sigmoid(self.fc4(h))

        ###################################
        return o

    def forward(self, x):
        #glue the encoder and the decoder together
        h = self.encode(x)
        o = self.decode(h)
        return o

    def process(self, magnitude):
        #process the whole chunk of spectrogram at run time
        result= magnitude.copy()
        with torch.no_grad():
            nFrame = magnitude.shape[1]
            for i in range(nFrame):
                result[:,i] = magnitude[:,i]*self.forward(torch.from_numpy(magnitude[:,i].reshape(1,-1))).numpy()
        return result 

def validate(model, dataloader):
    model.eval()

    window = 20
    with torch.no_grad():
    #Each time fetch a batch of samples from the dataloader
        for sample in dataloader:

            mixturex = sample['mixture'].to(device)
            targetx = sample['vocal'].to(device)


            mixtureT = tensorTransform(mixturex,window)
            targetT = tensorTransform(targetx,window)
            loss_T = 0
            T = len(mixtureT)           
            for tau in range(T):
                out = model.forward(mixtureT[tau])

                pred_v = out*mixtureT[tau]
                loss = KL_loss(pred_v,targetT[tau])  
                loss_T += loss.item()
            
            #if lossMovingAveraged<0:
                #lossMovingAveraged = loss_T/(window*T) 
            #else:
                #lossMovingAveraged = eta * lossMovingAveraged+ (1-eta)*(loss_T/(window*T)) 

    validationLoss = loss_T/(window*T) 

    ######################################################################################
    # Implement here your validation loop. It should be similar to your train loop
    # without the backpropagation steps
    ######################################################################################


    model.train()
    return validationLoss
def tensorTransform(Tensor,colStack):
    """
    Transposes the Matrix and selects the given
    number of column.
    Returns a list of tensors for iteration
    """
    TensorT = torch.transpose(Tensor,1,2)
    i =0
    TensorTlist = []
    while i<TensorT.shape[1]:
        TensorT_col = TensorT[:,i:i+colStack,:]
        TensorTlist.append(TensorT_col)
        i+=colStack

    return TensorTlist

def KL_loss(out,target):
    loss = torch.sum(torch.mean(target*(torch.log(target+1e-4) - torch.log(out+ 1e-4)) -target+out,dim=1))

    return loss



def saveFigure(result, target, mixture):
    plt.subplot(3,1,1)
    plt.pcolormesh(np.log(1e-4+result), vmin=-300/20, vmax = 10/20)
    plt.title('estimated')

    plt.subplot(3,1,2)
    plt.pcolormesh(np.log(1e-4+target.cpu()[0,:,:].numpy()), vmin=-300/20, vmax =10/20)
    plt.title('vocal')
    plt.subplot(3,1,3)

    plt.pcolormesh(np.log(1e-4+mixture.cpu()[0,:,:].numpy()), vmin=-300/20, vmax = 10/20)
    plt.title('mixture')

    plt.savefig("result_feedforward.png")
    plt.gcf().clear()

if __name__ == "__main__":
    ######################################################################################
    # Load Args and Params
    ######################################################################################
    parser = argparse.ArgumentParser(description='Train Arguments')
    parser.add_argument("--blockSize", type=int, default = 4096)
    parser.add_argument('--hopSize', type=int, default = 2048)
    # how many audio files to process fetched at each time, modify it if OOM error
    parser.add_argument('--batchSize', type=int, default = 8)
    # set the learning rate, default value is 0.0001
    parser.add_argument('--lr', type=float, default=1e-4)
    # Path to the dataset, modify it accordingly
    parser.add_argument('--dataset', type=str, default = '../DSD100')
    # set --load to 1, if you want to restore weights from a previous trained model
    parser.add_argument('--load', type=int, default = 0)
    # path of the checkpoint that you want to restore
    parser.add_argument('--checkpoint', type=str, default = 'savedModel_feedForward_best.pt')

    parser.add_argument('--seed', type=int, default = 555)
    args = parser.parse_args()

    # Random seeds, for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    fs = 32000
    blockSize = args.blockSize
    hopSize = args.hopSize
    PATH_DATASET = args.dataset
    batchSize= args.batchSize
    minValLoss = np.inf

    # transformation pipeline for training data
    transformTrain = transforms.Compose([
        #Randomly rescale the training data
        Data.Transforms.Rescale(0.8, 1.2),

        #Randomly shift the beginning of the training data, because we always do chunking for training in this case
        Data.Transforms.RandomShift(fs*30),

        #transform the raw audio into spectrogram
        Data.Transforms.MakeMagnitudeSpectrum(blockSize = blockSize, hopSize = hopSize),

        #shuffle all frames of a song for training the single-frame model , remove this line for training a temporal sequence model
        Data.Transforms.ShuffleFrameOrder()
    ])

    # transformation pipeline for training data. Here, we don't have to use any augmentation/regularization techqniques
    transformVal = transforms.Compose([
        #transform the raw audio into spectrogram
        Data.Transforms.MakeMagnitudeSpectrum(blockSize = blockSize, hopSize = hopSize),
    ])

    #initialize dataloaders for training and validation data, every sample loaded will go thourgh the preprocessing pipeline defined by the above transformations
    #workers will restart after each epoch, which takes a lot of time. repetition = 8  repeats the dataset 8 times in order to reduce the waiting time
    # so, in this case,  1 epoch is equal to 8 epochs. For validation data, there is not point in repeating the dataset.
    datasetTrain = Data.DSD100Dataset(PATH_DATASET, split = 'Train', mono =True, transform = transformTrain, repetition = 8)
    datasetValid = Data.DSD100Dataset(PATH_DATASET, split = 'Valid', mono =True, transform = transformVal, repetition = 1)

    #initialize the data loader
    #num_workers means how many workers are used to prefetch the data, reduce num_workers if OOM error
    dataloaderTrain = torch.utils.data.DataLoader(datasetTrain, batch_size = batchSize, shuffle=True, num_workers = 2, collate_fn = Data.collate_fn)
    dataloaderValid = torch.utils.data.DataLoader(datasetValid, batch_size = 10, shuffle=False, num_workers = 0, collate_fn = Data.collate_fn)

    #initialize the Model
    model = ModelSingleStep(blockSize)

    # if you want to restore your previous saved model, set --load argument to 1
    if args.load == 1:
        checkpoint = torch.load(args.checkpoint)
        minValLoss = checkpoint['minValLoss']
        model.load_state_dict(checkpoint['state_dict'])


    #determine if cuda is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    #initialize the optimizer for paramters
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)

    model.train(mode=True)

    lossMovingAveraged  = -1

    #################################### 
    #The main loop of training
    #################################### 
    eta = 0.99
    for epoc in range(100):
        iterator = iter(dataloaderTrain)
        with trange(len(dataloaderTrain)) as t:
            for idx in t:
                #Each time fetch a batch of samples from the dataloader
                sample = next(iterator)
                #the progress of training in the current epoch


                #read the input and the fitting target into the device
                mixture = sample['mixture'].to(device)
                target = sample['vocal'].to(device)

                seqLen = mixture.shape[2]
                winLen = mixture.shape[1]
                currentBatchSize = mixture.shape[0]

                #store the result for the first one for debugging purpose
                result = torch.zeros((winLen, seqLen), dtype=torch.float32)

                                #################################
                                #################################
                # taking smaller steps as we go deeper into training 
                #if epoc == 50:
                    #optimizer = torch.optim.Adam(model.parameters(), lr = args.lr/5)
                #elif epoc ==70:
                    #optimizer = torch.optim.Adam(model.parameters(), lr = args.lr/10)
                #elif epoc == 90:
                    #optimizer = torch.optim.Adam(model.parameters(), lr = args.lr/20)


                # iterating through each T (column) in STFT of audio file

                #transposing the tensors to get columns
                window = 20
                mixtureT = tensorTransform(mixture,window)
                targetT = tensorTransform(target,window)

                T = len(mixtureT)
                mask = [] 
                loss_T = 0
                for tau in range(T):
                    model.zero_grad()

                    out = model.forward(mixtureT[tau])
                    

                    pred_v = out*mixtureT[tau]
                    loss = KL_loss(pred_v,targetT[tau])
                    loss.backward()
                    optimizer.step()
                    loss_T += loss.item()       

                # store your smoothed loss here
                if lossMovingAveraged <0:
                    lossMovingAveraged = loss_T/seqLen
                else:
                    lossMovingAveraged = eta * lossMovingAveraged+ (1-eta)*(loss_T/seqLen)
                # this is used to set a description in the tqdm progress bar 
                t.set_description(f"epoc : {epoc}, loss {lossMovingAveraged}")
                #save the model

            # plot the first one in the batch for debuging purpose
            saveFigure(result, target, mixture)

        # create a checkpoint of the current state of training
        checkpoint = {
            'state_dict': model.state_dict(),
            'minValLoss': minValLoss,
        }
        # save the last checkpoint
        torch.save(checkpoint, 'savedModel_feedForward_last.pt')

        #### Calculate validation loss
        valLoss = validate(model, dataloaderValid)
        print(f"validation Loss = {valLoss:.4f}")

        if valLoss < minValLoss:
            minValLoss = valLoss
            # then save checkpoint
            checkpoint = {
                'state_dict': model.state_dict(),
                'minValLoss': minValLoss,
            }
            torch.save(checkpoint, 'savedModel_feedForward_best.pt')



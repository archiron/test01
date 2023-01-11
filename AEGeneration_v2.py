#!/usr/bin/env python
# coding: utf-8

################################################################################
# AEGeneration : create a KS comparison (max diff) between the original curve 
# and the predicted one for different egamma validation releases.
#
# MUST be launched with the cmsenv cmd after a cmsrel cmd !!
#                                                                              
# Arnaud Chiron-Turlay LLR - arnaud.chiron@llr.in2p3.fr                        
#                                                                              
################################################################################

import time
import sys, os
import importlib
import importlib.machinery
import importlib.util
from pathlib import Path

#import seaborn # only with cmsenv on cca.in2p3.fr

# lines below are only for func_Extract
from sys import argv

if len(sys.argv) > 1:
    print(sys.argv)
    print("step 4 - arg. 0 :", sys.argv[0]) # name of the script
    print("step 4 - arg. 1 :", sys.argv[1]) # COMMON files path
    print("step 4 - arg. 2 :", sys.argv[2]) # FileName for paths
    commonPath = sys.argv[1]
    filePaths = sys.argv[2]
    workPath=sys.argv[1][:-12]
else:
    print("rien")
    resultPath = ''

import pandas as pd
import numpy as np
## WARNING pbm with torch
import torch
from torch.utils import data
from torch.nn.functional import normalize

# these line for daltonians !
#seaborn.set_palette('colorblind')

print("\nAE Generation")

# Import module
loader = importlib.machinery.SourceFileLoader( filePaths, commonPath+filePaths )
spec = importlib.util.spec_from_loader( filePaths, loader )
blo = importlib.util.module_from_spec( spec )
loader.exec_module( blo )
print('DATA_SOURCE : %s' % blo.DATA_SOURCE)
resultPath = blo.RESULTFOLDER 
print('result path : {:s}'.format(resultPath))

Chilib_path = workPath + '/' + blo.LIB_SOURCE # checkFolderName(blo.LIB_SOURCE) # sys.argv[1]
print('Lib path : {:s}'.format(Chilib_path))
sys.path.append(Chilib_path)
sys.path.append(commonPath)

import default as dfo
from default import *
from rootValues import NB_EVTS
from defaultStd import *
from autoEncoders import *
from controlFunctions import *
from graphicAutoEncoderFunctions import *

from DecisionBox import *

arrayKSValues = []
rels = []

# get the branches for ElectronMcSignalHistos.txt
branches = []
source = Chilib_path + "/HistosConfigFiles/ElectronMcSignalHistos.txt"
branches = getBranches(tp_1, source)
cleanBranches(branches) # remove some histo wich have a pbm with KS.

nbBranches = len(branches) # [0:8]
print('there is {:03d} datasets'.format(nbBranches))

resultPath += '/' + str(NB_EVTS)
resultPath = checkFolderName(resultPath)
print('resultPath : {:s}'.format(resultPath))

# get list of generated ROOT files
rootPath = "/data_CMS/cms/chiron/ROOT_Files/CMSSW_12_5_0_pre4/"
print('rootPath : {:s}'.format(rootPath))
rootFilesList_0 = getListFiles(rootPath, 'root')
print('there is ' + '{:03d}'.format(len(rootFilesList_0)) + ' generated ROOT files')
nbFiles = change_nbFiles(len(rootFilesList_0), nbFiles)

folder = resultPath + checkFolderName(dfo.folder)
data_dir = folder + '/{:03d}'.format(nbFiles)
print('data_dir path : {:s}'.format(data_dir))
data_res = data_dir + '/AE_RESULTS/'
print('data_res path : {:s}'.format(data_res))

# get list of added ROOT files
rootFolderName = workPath + '/' + blo.DATA_SOURCE # '/pbs/home/c/chiron/private/KS_Tools/GenExtract/DATA/NewFiles'
print('rootFolderName : {:s}'.format(rootFolderName))
rootFilesList = getListFiles(rootFolderName, 'root')
print('there is ' + '{:03d}'.format(len(rootFilesList)) + ' added ROOT files')
for item in rootFilesList:
    #print('%s' % item)
    b = (item.split('__')[2]).split('-')
    rels.append([b[0], b[0][6:]])
sortedRels = sorted(rels, key = lambda x: x[0]) # gives an array with releases sorted
for elem in sortedRels:
    print(elem)

# get list of text files
pathKSFiles = data_dir
print('KS path : %s' % pathKSFiles)
KSlistFiles = []
tmp = getListFiles(pathKSFiles, 'txt')
for elem in tmp:
    if (elem[5:10] == '_diff'): # to keep only histo_differences_KScurves files
        KSlistFiles.append(elem)
print(KSlistFiles, len(KSlistFiles))
    
for item in KSlistFiles:
    print('file : %s' % item)
    aa = item.split('__')[0]
    fileName = pathKSFiles + '/' + item
    file1 = open(fileName, 'r')
    bb = file1.readlines()
    for elem in bb:
        tmp = []
        cc = elem.split(' : ')
        tmp = [cc[0], aa, float(cc[1][:-1])]
        arrayKSValues.append(tmp)
sortedArrayKSValues = sorted(arrayKSValues, key = lambda x: x[0]) # gives an array with releases sorted
for elem in sortedArrayKSValues:
    print("sortedArrayKSValues", elem)

if (len(KSlistFiles) != len(rootFilesList)):
    print('you must have the same number of KS files than releases')
    exit()
else:
    print('we have the same number of KS files than releases')

#load data from branchesHistos_NewFiles.txt file ..
fileName = data_dir + "/branchesHistos_NewFiles.txt"
print('%s' % fileName)
file1 = open(fileName, 'r')
Lines = file1.readlines()

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print('device : {:s}'.format(str(device)))
if torch.cuda.is_available():
    device = torch.device("cuda:3")
    use_GPU = True
else:
    device = torch.device("cpu")
print('device : {:s}'.format(str(device)))

timeFolder = time.strftime("%Y%m%d-%H%M%S")

folderName = data_res + createAEfolderName(hidden_size_1, hidden_size_2, hidden_size_3, hidden_size_4, useHL3, useHL4, latent_size) # , timeFolder, nbFiles, branches[i]
checkFolder(folderName)
print('\nComplete folder name : {:s}'.format(folderName))

loopMaxValue = 10 # nbBranches #25 # nbBranches
for i in range(0, loopMaxValue):
    print('{:s}\n'.format(branches[i]))
    df = []
    fileName = resultPath + "/histo_" + branches[i] + '_{:03d}'.format(nbFiles) + ".txt"
    if Path(fileName).exists():
        print('{:s} exist'.format(fileName))
        df = pd.read_csv(fileName)
    else:
        print('{:s} does not exist'.format(fileName))
        continue

    # add a subfolder with the name of the histo and a folder with date/time
    folderNameBranch = folderName + branches[i] + '/' + timeFolder
    checkFolder(folderNameBranch)
    print('\n===== folderNameBranch : {:s} ====='.format(folderNameBranch))

    df_entries = []
    linOp = []

    for line in Lines:
        rel,b = line.rstrip().split(',', 1)
        hName = b.rstrip().split(',', 1)[0]
        if ( str(hName) == str(branches[i])):
            linOp.append(line)
            print(line)

    torch_tensor_entries = []
    torch_tensor_entries_n = []

    train_loader = []
    test_loader = []

    # add a subfolder for the losses
    folderNameLosses = folderNameBranch + '/Losses/'
    checkFolder(folderNameLosses)
    print('\nfolderNameLosses : {:s}'.format(folderNameLosses))

    tmp = df 
    cols = df.columns.values
    cols_entries = cols[6::2]
    df_entries = tmp[cols_entries]
    (_, Ncols) = df_entries.shape

    # get nb of columns & rows for histos & remove over/underflow
    (Nrows, Ncols) = df_entries.shape
    print('before : \t[Nrows, Ncols] : [%3d, %3d] for %s' % (Nrows, Ncols, branches[i]))
    df_entries = df_entries.iloc[:, 1:Ncols-1]
    (Nrows, Ncols) = df_entries.shape
    print('after : \t[Nrows, Ncols] : [%3d, %3d] for %s' % (Nrows, Ncols, branches[i]))
    #print(df_entries)

    # add a subfolder for the losses
    folderNameLoader = folderNameBranch + '/TrainTestLOADER/'
    checkFolder(folderNameLoader)
    print('\nfolderNameLoader : {:s}'.format(folderNameLoader))

    trainName = folderNameLoader + "multi_train_loader_" + branches[i] + "_{:03d}".format(nbFiles) + ".pth"
    testName = folderNameLoader + "multi_test_loader_" + branches[i] + "_{:03d}".format(nbFiles) + ".pth"

    if (useTrainLoader == 1):
        tmpPath = folderName + branches[i] + '/' + TimeFolderRef + '/TrainTestLOADER/'
        trainName = tmpPath + "multi_train_loader_" + branches[i] + "_{:03d}".format(nbFiles) + ".pth"
        testName = tmpPath + "multi_test_loader_" + branches[i] + "_{:03d}".format(nbFiles) + ".pth"
        print('load %s.' % trainName)
        print('load %s.' % testName)
        if not os.path.isfile(trainName):
            print('%s does not exist' % trainName)
            exit()
        else:
            encoder = torch.load(trainName)
            train_loader = torch.load(trainName)
        if not os.path.isfile(testName):
            print('%s does not exist' % testName)
            exit()
        else:
            test_loader = torch.load(testName)
        print('load OK.')
    else:
        # creating torch tensor from df_entries/errors
        torch_tensor_entries = torch.tensor(df_entries.values,device=device) # 
        # normalize the tensor
        torch_tensor_entries_n = normalize(torch_tensor_entries, p=2.0)

        train_size=int(percentageTrain*len(torch_tensor_entries)) # in general torch_tensor_entries = 200
        test_size=len(torch_tensor_entries)-train_size
        print('%d : train size : %d' % (i,train_size))
        print('%d : test size  : %d' % (i,test_size))
        train_tmp, test_tmp = data.random_split(torch_tensor_entries_n,[train_size,test_size])
    
        train_loader = data.DataLoader(train_tmp,batch_size=batch_size)
        test_loader = data.DataLoader(test_tmp,batch_size=batch_size)
        
        print('saving ... %s' % trainName)
        torch.save(train_loader,trainName)
        torch.save(test_loader,testName)
        print('save OK.\n')

    if use_GPU:
        loss_fn = torch.nn.MSELoss().cuda()
    else:
        loss_fn=torch.nn.MSELoss()

    #define the network
    if useHL4 == 1:
        encoder=Encoder4(device,latent_size,Ncols,hidden_size_1,hidden_size_2,hidden_size_3,hidden_size_4)
        decoder=Decoder4(device,latent_size,Ncols,hidden_size_1,hidden_size_2,hidden_size_3,hidden_size_4)
        nbLayer = 4
    elif useHL3 == 1:
        encoder=Encoder3(device,latent_size,Ncols,hidden_size_1,hidden_size_2,hidden_size_3)
        decoder=Decoder3(device,latent_size,Ncols,hidden_size_1,hidden_size_2,hidden_size_3)
        nbLayer = 3
    else: # 2 layers
        encoder=Encoder2(device,latent_size,Ncols,hidden_size_1,hidden_size_2)
        decoder=Decoder2(device,latent_size,Ncols,hidden_size_1,hidden_size_2)
        nbLayer = 2

    encoder.to(device)
    decoder.to(device)

    params_to_optimize=[
    {'params': encoder.parameters()},
    {'params': decoder.parameters()}
    ]

    optim=torch.optim.Adam(params_to_optimize,lr=lr,weight_decay=1e-05)
    history_da={'train_loss':[],'test_loss':[]}
    L_out = []
    LatentValues_Train = []
    LatentValues_Test = []

    # Ready for calculation
    encoderName = folderNameLoader + "/mono_encoder_{:01d}_".format(nbLayer) + branches[i] + "_{:03d}".format(nbFiles) + ".pth"
    decoderName = folderNameLoader + "/mono_decoder_{:01d}_".format(nbLayer) + branches[i] + "_{:03d}".format(nbFiles) + ".pth"

    # add a subfolder for the pictures
    folderNamePict = folderNameBranch + '/Pictures/'
    checkFolder(folderNamePict)
    print('\nfolderNamePict : {:s}'.format(folderNamePict))

    lossesPictureName = folderNamePict + '/loss_plots_' + branches[i] + "_{:03d}".format(nbFiles) + '.png'
    if ( useEncoder == 1):
        if not os.path.isfile(encoderName):
            print('%s does not exist' % encoderName)
            exit()
        else:
            encoder = torch.load(encoderName)
        if not os.path.isfile(decoderName):
            print('%s does not exist' % decoderName)
            exit()
        else:
            decoder = torch.load(decoderName)
    else:
        for epoch in range(nb_epochs):
            train_loss, encoded_out=train_epoch_den(encoder=encoder, decoder=decoder,device=device,
                dataloader=train_loader, loss_fn=loss_fn,optimizer=optim)
            print('epoch : ', epoch, encoded_out.detach().numpy())
            test_loss, d_out, latent_out=test_epoch_den(encoder=encoder, decoder=decoder,device=device,
                dataloader=test_loader, loss_fn=loss_fn)
            L_out.append(d_out)
            LatentValues_Train.append(encoded_out.detach().numpy())
            LatentValues_Test.append(latent_out)
            history_da['train_loss'].append(train_loss)
            history_da['test_loss'].append(test_loss)

        r = (train_loss - test_loss) / (train_loss + test_loss)
        print('epoch : %03d : tr_lo = %e : te_lo = %e : r = %e' % (epoch, train_loss, test_loss, r))
        #print('epoch : %03d : tr_lo = %e : te_lo = %e' % (epoch, train_loss, test_loss))
        if ( saveEncoder == 1 ): # warning encoder & decoder are needed for next computations
            torch.save(encoder,encoderName)
            torch.save(decoder,decoderName)

        #print('write HL_1 : %03d, HL_2 : %03d, LT : %03d :: tr_loss : %e, te_loss : %e'
        #            % (hidden_size_1, hidden_size_2, latent_size, train_loss, test_loss))

        #createLossPictures(branches[i], history_da, nb_epochs, lossesPictureName)
        createLossPictures(branches[i], history_da, epoch+1, lossesPictureName)

        labels_Train = []
        labels_Test = []
        x_Train = []
        y_Train = []
        x_Test = []
        y_Test = []
        title='Train/Test latent picture in 2 dim'
        pictureName = folderNamePict + '/traintestLatentPicture_' + branches[i] + '.png'
        for ind in range(0, len(LatentValues_Train)):
            #print('Train ', ind, LatentValues_Train[ind])
            x_Train.append(LatentValues_Train[ind][0])
            y_Train.append(LatentValues_Train[ind][1])
            labels_Train.append(i)
        
        for ind in range(0, len(LatentValues_Test)):
            #print('Test ', ind, LatentValues_Test[ind][0])
            x_Test.append(LatentValues_Test[ind][0].numpy()[0])
            y_Test.append(LatentValues_Test[ind][0].numpy()[1])
            labels_Test.append(i)
        print('createLatentPictureTrainTest call')
        createLatentPictureTrainTest(x_Train,y_Train,x_Test,y_Test, pictureName, title)
        #createLatentPictureTrainTest(x_Test,y_Test,x_Test,y_Test, pictureName2, title)

        nb_history_da = len(history_da['train_loss'])
        t1 = np.asarray(history_da['train_loss'])
        t2 = np.asarray(history_da['test_loss'])
        rr = 1.
        ss = 1.
        for kk in range(nb_history_da-10,nb_history_da):
            rr *= t1[kk]/t2[kk]
            ss *= t2[kk]/t1[kk]
        print('coefficient losses : {:1.4e}'.format(rr) + ' - {:1.4e}'.format(ss))
    
    # Ready for prediction
    print('using %s\n' % encoderName)

    predLossesValues = folderNameBranch + "/predLossesValues_" + branches[i] + ".txt"
    print("loss values file : %s" % predLossesValues)
    wPred = open(predLossesValues, 'w')

    # export the y_pred_new values
    predValues = folderNameBranch + "/predValues_" + branches[i] + ".txt"
    print("values file : %s" % predValues)
    wPredVal = open(predValues, 'w')

    lossesVal = []
    latentVal = []
    LinesPred = []
    for elem in linOp:
        print('linOp : ', elem)
        rel, hName,line = elem.rstrip().split(',', 2)
        #print(rel,hName,line)
        new = line.rstrip().split(',')
        new = np.asarray(new).astype(float)
        #print(new)
        df_new = pd.DataFrame(new).T # otherwise, one column with 50 rows instead of 1 line with 50 columns

        # creating torch tensor from df_entries/errors
        torch_tensor_new = torch.tensor(df_new.values,device=device)

        # normalize the tensor
        torch_tensor_entries_n = normalize(torch_tensor_new, p=2.0)
        test_loader_n = data.DataLoader(torch_tensor_entries_n)

        encoder = torch.load(encoderName)
        decoder = torch.load(decoderName)
        loss_fn = torch.nn.MSELoss()

        params_to_optimize=[
        {'params': encoder.parameters()},
        {'params': decoder.parameters()}
        ]

        optim=torch.optim.Adam(params_to_optimize,lr=lr,weight_decay=1e-05)

        # Forward pass: Compute predicted y by passing x to the model
        new_loss, y_pred_new, latent_out = test_epoch_den(encoder=encoder,
                decoder=decoder,device=device,
                dataloader=test_loader_n,
                loss_fn=loss_fn)
        #print(new_loss)
        #print(y_pred_new)
        #print(latent_out)

        # Compute and print loss
        wPred.write('%e, %s\n' % (new_loss, rel))
        lossesVal.append([rel,new_loss.item()])
        latentVal.append(latent_out[0].numpy())
        #print(torch_tensor_entries_n)
        #print(y_pred_new)

        pictureName = folderNamePict + '/predicted_new_curves_' + branches[i] + '_' + rel[6:] + '_multi.png'
        ### WARNING rel is the same for all comparisons !!!
        creatPredPictLinLog(branches[i], Ncols, torch_tensor_entries_n, y_pred_new, new_loss, rel[6:], pictureName)

        # write values into the predValues file (# export the y_pred_new values)
        text2write = rel + ',' + branches[i]
        for val in y_pred_new.numpy():
            N=len(val)
            for nn in range(0,N):
                text2write += ',' + str(val[nn])
        LinesPred.append(text2write)
        text2write += '\n'
        wPredVal.write(text2write)

    labels = []
    val = []
    x = []

    sortedLossesVal = sorted(lossesVal, key = lambda x: x[0])
    for elem in sortedLossesVal:
        labels.append(elem[0][6:])
        val.append(elem[1])

    pictureName = folderNamePict + '/comparison_loss_values_' + branches[i] + '.png'
    title = r"$\bf{" + branches[i] + "}$" + ' : Comparison of the losses values as function of releases.'
    createCompLossesPicture(labels,val, pictureName, title)

print('end')


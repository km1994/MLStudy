import sys
sys.path.append('../')
############################
## Network Parameters     ##
############################

#Input Size
height = 28
width = 28
channels = 1

#Layers
layers = [ ('conv',6, 5, 1), ('pool',2,2), ('conv',16,5, 1 ),('pool',2,2),('fc',120),('fc',84), ('softmax',10)  ]

activation = 'relu'
pool = 'max' # 'mean' or 'max'

#Network Initialisation
initBias = 0.01   # Initial Bias Value for all layers

###########################
## Training Parameters   ##
###########################

alpha = 0.9 # Momentum
lr = 0.01
numEpoch = 1000
batchSize = 1
trainExamples = 1
validate = True
valExamples = 500
pretrain = False
trainedModel = "../models/1_0.01_model.mat"

###########################
## Save Models           ##
###########################
logDirectory = "logs/"
log = True
trainlog = logDirectory+ str(batchSize) +"_"+ str(lr) + "_train.log"
vallog =  logDirectory+str(batchSize) +"_"+ str(lr) + "_val.log"
saveModel = True
modelDirectory = "models/"
modelFile = modelDirectory + str(batchSize) + '_'+ str(lr) + '_model.mat'

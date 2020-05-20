import numpy as np
import pickle

#this file will do
#1. read from list of .pkl files that will become training data
#2. save wMean.dat and wSd.dat

for input in ["inputHeart/","inputBow/","inputAcrobat/"]:

	fileList=[
		'data_train.pkl',
	]

	collect=[]
	for f in fileList:
		#data=np.load('input/'+f) #(2,-)
		dataList=pickle.load(open(input+f,'rb'))
		for data in dataList:
			collect.append(data)

	allTrainingData=np.hstack(collect)
	mean=np.mean(allTrainingData, axis=1, keepdims=True)
	sd=np.std(allTrainingData-mean, axis=1, keepdims=True)

	mean.dump(input+'wMean.dat')    #(2,1)
	sd.dump(input+'wSD.dat')    #(2,1)

print("done")

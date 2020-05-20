import numpy as np
import os
import pickle
#from matplotlib import pyplot as plt
import random
#import dataGenerator

def heartTransform(t):
	return 1.8*np.sin(t)**3, 1.3*np.cos(t)-0.5*np.cos(2*t)-0.2*np.cos(3*t)-0.1*np.cos(4*t)

def bowTransform(t):
	return 1.4*np.cos(t), 2.9*np.sin(t)*np.cos(t)

def acrobatTransform(t):
	return 1.6*np.cos(0.5*t)*(np.sin(0.5*t+0.5)+np.cos(1.5*t+0.5)), 1.6*np.sin(0.5*t)*(np.sin(0.5*t+0.5)-np.cos(1.5*t+0.5))

for shape in range(3):

	if(shape==0):
		transformer=heartTransform
		input="inputHeart/"
	elif(shape==1):
		transformer=bowTransform
		input="inputBow/"
	elif(shape==2):
		transformer=acrobatTransform
		input="inputAcrobat/"

	for gen in ['train','test']:
		if(gen=='train'):
			np.random.seed(2)
		elif(gen=='test'):
			np.random.seed(3)

		os.makedirs(input,exist_ok=True)

		#plt.figure()

		dataList=[]
		phaseGroundtruthList=[]

		for j in range(1):
			

			dl=15000
			dt=np.zeros([dl])
			randomProgress=np.random.uniform(0*np.pi/100,4*np.pi/100,dl)
			randomX=np.random.normal(scale=0.05,size=dl)
			randomY=np.random.normal(scale=0.05,size=dl)
			
			for i in range(1,dl):
				dt[i]=dt[i-1]+randomProgress[i]

			dx,dy=transformer(dt)
			
			dx+=randomX
			dy+=randomY

			#plt.plot(dx,dy)
			#plt.plot(np.arange(2000),dx[10:2010])
			#plt.plot(np.arange(2000),dt[10:2010]%(2*np.pi))

			dataList.append(np.stack([dx,dy],axis=0))
			phaseGroundtruthList.append(dt)

		pickle.dump(dataList,open(input+"data_"+gen+".pkl",'wb'))
		if(gen=='test'):
			pickle.dump(phaseGroundtruthList,open(input+"data_"+gen+"_phaseGroundTruth.pkl",'wb'))

		#plt.axis('equal')
		#plt.grid()
		#plt.show()


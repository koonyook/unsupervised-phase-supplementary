import tensorflow as tf
import numpy as np
import random
import os,sys
import pickle
import util

import matplotlib
if os.name != 'nt':
	matplotlib.use('Agg') 
import matplotlib.pyplot as plt


if(len(sys.argv)==4):
	if(sys.argv[1]=='learn'):
		runTraining=True
	elif(sys.argv[1]=='infer'):
		runTraining=False
	else:
		print("The first argument must be 'learn' or 'infer'.")

	if(sys.argv[2]=='heart'):
		input='inputHeart/'
	elif(sys.argv[2]=='bow'):
		input='inputBow/'
	elif(sys.argv[2]=='acrobat'):
		input='inputAcrobat/'
	else:
		print("Incorrect shape argument.")
	
	folder=sys.argv[3]+'/'

else:
	print('Incorrect arguments')
	exit()

useMeanSd=True
sphereCount=0

saveDataForAnimation=False
animationDataInterval=1	#1=every step

seed=0

gpuID='0'

fileList=[
    'data_train.pkl',
]

testFileList=[
	'data_test.pkl',
]

initFolder = folder+'init/'

h=30
hiddenLayers=[h,h,h,h,h]

#profile2 (core)
Cp = (-180 + 45)*np.pi/180	#on the left
Ap=0.0*np.pi/180	
Bp=4.0*np.pi/180

centerSigma=1
distWeight=1
singWeight=1

iterationToDo=5000

cropHead=0
cropTail=0
regressWing=7	
predictionGap=1	 #this might be important in very high sampling rate scenario

windowWing=3
windowSize=1+windowWing*2

sessionName="learnPhase.ckpt"
sessionPath=folder+sessionName

logDirectory=folder+'/logGD/'
imageFolder=folder+'/img/'

if not os.path.exists(folder+'checkpoint'):
	os.makedirs(folder,exist_ok=True)
	genNewData=True
	startNewTraining=True
else:
	genNewData=False
	startNewTraining=False

if not os.path.exists(logDirectory):
	os.makedirs(logDirectory)

if not os.path.exists(imageFolder):
	os.makedirs(imageFolder)

recordList = []
testRecordList=[]
for inputFileName in fileList:
	aList=pickle.load(open(input+inputFileName,'rb'))
	for aRecord in aList:
		#print(aRecord.shape)	#(2,?)
		aRecord=aRecord[:,cropHead:aRecord.shape[1]-cropTail]
		recordList.append(aRecord)

for inputFileName in testFileList:
	aList=pickle.load(open(input+inputFileName,'rb'))
	for aRecord in aList:
		#print(aRecord.shape)	#(2,?)
		aRecord=aRecord[:,cropHead:aRecord.shape[1]-cropTail]
		testRecordList.append(aRecord)

D=recordList[0].shape[0]
#print(D)

scaledList = []
testScaledList = []
if(useMeanSd):
	wMean=np.load(input+'wMean.dat',allow_pickle=True)	#(2,1)
	wSD=np.load(input+'wSD.dat',allow_pickle=True)	#(2,1)
	scale=1/wSD	#change variance to 1
	for aRecord in recordList:
		scaledList.append((aRecord-wMean)*scale) 	

	for aRecord in testRecordList:
		testScaledList.append((aRecord-wMean)*scale) 	

else: 
	for aRecord in recordList:
		scaledList.append(aRecord*1.0) 	#I don't care about scaling now

nowPoseList=[]
nextPoseList=[]
nowHeadList=[]
nextHeadList=[]

metaDict={'seriesBeginIndex':[],'animationDataInterval':animationDataInterval}

for aScaled in scaledList:
	m=aScaled.shape[1]
	allPose=aScaled[:,regressWing:m-regressWing]	#(D,m-2*regressWing)
	allHead=util.getRegressHeadingDirectionWithSphere(aScaled, sphereCount, regressWing=regressWing)	#(D,m-2*regressWing)

	#generate all possible window
	allPoseWindow=[]
	allHeadWindow=[]
	for i in range(m-2*regressWing-windowSize+1):
		allPoseWindow.append(allPose[:,i:i+windowSize])
		allHeadWindow.append(allHead[:,i:i+windowSize])
	
	metaDict['seriesBeginIndex'].append(len(nowPoseList))

	nowPoseList+=allPoseWindow[:-predictionGap]
	nextPoseList+=allPoseWindow[predictionGap:]
	nowHeadList+=allHeadWindow[:-predictionGap]
	nextHeadList+=allHeadWindow[predictionGap:]

metaDict['pairN']=len(nowPoseList)

pairCount=len(nowPoseList)
nowState=np.zeros([2*D*windowSize,pairCount]) #first half
nextState=np.zeros([2*D*windowSize,pairCount]) #second half

centerPose=np.zeros([2,pairCount])

for i in range(pairCount):
	nowState[:,i]=np.concatenate([nowPoseList[i].flatten(),nowHeadList[i].flatten()])
	nextState[:,i]=np.concatenate([nextPoseList[i].flatten(),nextHeadList[i].flatten()])
	
	centerPose[:,i]=nowPoseList[i][:,windowSize//2]

metaDict['centerPose']=centerPose

#prepare feedDict for training
firstLayerFeed=np.hstack([nowState,nextState])


######################################
#prepare test data
testNowPoseList=[]
testNowHeadList=[]
for aScaled in testScaledList:
	m=aScaled.shape[1]
	allPose=aScaled[:,regressWing:m-regressWing]	#(D,m-2*regressWing)
	allHead=util.getRegressHeadingDirectionWithSphere(aScaled, sphereCount, regressWing=regressWing)	#(D,m-2*regressWing)

	#generate all possible window
	allPoseWindow=[]
	allHeadWindow=[]
	for i in range(m-2*regressWing-windowSize+1):
		allPoseWindow.append(allPose[:,i:i+windowSize])
		allHeadWindow.append(allHead[:,i:i+windowSize])
	
	testNowPoseList+=allPoseWindow
	testNowHeadList+=allHeadWindow

testCount=len(testNowPoseList)
testNowState=np.zeros([2*D*windowSize,testCount])
testCenterPose=np.zeros([2,testCount])

for i in range(testCount):
	testNowState[:,i]=np.concatenate([testNowPoseList[i].flatten(),testNowHeadList[i].flatten()])
	testCenterPose[:,i]=testNowPoseList[i][:,windowSize//2]


##########################################
###########define calculation graph#######
##########################################

with tf.device('/gpu:'+gpuID):	
	config = tf.ConfigProto(allow_soft_placement=True)
	config.gpu_options.allow_growth=True
	sess = tf.Session(config=config)

	tf.set_random_seed(seed)

	inputSize=2*D*windowSize
	outputSize=2	#2 for x and y (before normalized)

	#placeholder
	firstLayerPlace = tf.placeholder(tf.float32, shape=[inputSize, None], name="input")

	global_step_tensor = tf.Variable(0, trainable=False, name='global_step')

	L=[inputSize]+hiddenLayers+[outputSize]	#60,-,-,-,2
	print(L)

	#name is important if you want to save the session
	W=[0]	#keep variable, W[1] should be W1 (no W0)
	b=[0]	#keep variable, b[1] should be b1 (no b0)

	for i in range(1,len(L)):
		Wi=tf.Variable(tf.random_normal([L[i],L[i-1]], 0, np.sqrt(1.0/float(L[i-1])),seed=seed),name="W"+str(i))
		bi = tf.Variable(tf.zeros([L[i],1]),name="b"+str(i))
		W.append(Wi)
		b.append(bi)

	A=[firstLayerPlace]
	for i in range(1,len(L)-1):
		Ai=tf.nn.tanh(tf.matmul(W[i],A[i-1])+b[i],name="A"+str(i))	
		A.append(Ai)

	i=len(L)-1
	A.append(tf.matmul(W[i],A[i-1])+b[i])	#linear for last layer

	prePhase=A[i]

	phaseXY=tf.nn.l2_normalize(prePhase,axis=0)	#this is a unit circle

	phaseRad=tf.atan2(phaseXY[1,:],phaseXY[0,:],name="phaseRad")	#use for output only

	#calculate penalty
	currentPhaseXY,nextPhaseXY=tf.split(phaseXY,2,axis=1)	#split into first 

	crossProductZ=tf.multiply(currentPhaseXY[0,:],nextPhaseXY[1,:])-tf.multiply(currentPhaseXY[1,:],nextPhaseXY[0,:])	#positive=counter-clockwise
	dotProduct=tf.reduce_sum(tf.multiply(currentPhaseXY,nextPhaseXY),axis=0)
	phaseProgress=tf.atan2(crossProductZ,dotProduct)

	#this is for 3 parameters (Cp, Ap, Bp)
	isInCA=tf.logical_and(tf.greater_equal(phaseProgress,Cp),tf.less(phaseProgress,Ap))	#range [Cp,Ap)
	y1=0
	y2=0.5*np.pi
	x1=Cp
	x2=Ap
	CA_penalty=tf.cos((phaseProgress*(y2-y1)+y1*x2-y2*x1)/(x2-x1))

	isInAB=tf.logical_and(tf.greater_equal(phaseProgress,Ap),tf.less_equal(phaseProgress,Bp))	#range [Ap,Bp]
	AB_penalty=tf.fill(tf.shape(phaseProgress), 0.0)

	#from B to C 
	y1=-0.5*np.pi  
	y2=0 
	x1=Bp
	x2=2*np.pi+Cp
	aboveB_penalty=tf.cos((phaseProgress*(y2-y1)+y1*x2-y2*x1)/(x2-x1))
	belowC_penalty=tf.cos(((phaseProgress+2*np.pi)*(y2-y1)+y1*x2-y2*x1)/(x2-x1))  #below C is shifted to be above 180

	isAboveB=tf.greater(phaseProgress,Bp)

	speedPenalty=tf.reduce_mean(tf.where(isInCA,CA_penalty,tf.where(isInAB,AB_penalty,tf.where(isAboveB,aboveB_penalty, belowC_penalty))))

	#backwardPenalty=tf.reduce_mean(tf.nn.relu(-(crossProductZ-minimumPhaseProgressRad)))

	badDistributionPenalty=distWeight*tf.reduce_sum(tf.square(tf.reduce_mean(phaseXY,axis=1)))	#1.0 is the worst

	singularityPenalty=singWeight*tf.reduce_mean((1/(centerSigma*np.sqrt(2*np.pi)))*tf.exp(-0.5*tf.square(tf.norm(prePhase,axis=0)/centerSigma)))

	cost=tf.add_n([speedPenalty,badDistributionPenalty,singularityPenalty],name='cost')

	optimizer = tf.train.AdamOptimizer()

	train = optimizer.minimize(cost,global_step=global_step_tensor, var_list=W[1:]+b[1:])
	saver = tf.train.Saver()

	if(runTraining):
		if(startNewTraining):
			sess.run(tf.global_variables_initializer())

			#save this initialization for repeatability
			os.makedirs(initFolder,exist_ok=True)
			saver.save(sess, initFolder+sessionName)

		else:
			saver.restore(sess, initFolder+sessionName)

		current_global_step=tf.train.global_step(sess, global_step_tensor)

		feedDict={firstLayerPlace:firstLayerFeed}
		print ('start cost', sess.run([cost,speedPenalty,badDistributionPenalty,singularityPenalty],feed_dict=feedDict))

		fig=plt.figure(figsize=(8, 8))	#reuse fig will save a lot of memory
		figRainbow=plt.figure(figsize=(8, 8))

		if(saveDataForAnimation):
			pickle.dump(metaDict,open(folder+"animationMetaDict.pkl",'wb'))
			animationDataFile=open(folder+"prePhaseData.bin","ab")	#append
		
		for step in range(current_global_step,current_global_step+iterationToDo+1):

			out = sess.run([train,cost,speedPenalty,badDistributionPenalty,singularityPenalty,prePhase,phaseXY,phaseRad],feed_dict=feedDict)

			prePhase_distribution,phaseXY_distribution,phaseOutput=out[5:]
			#get only first half (nowState)
			halfM=prePhase_distribution.shape[1]//2
			prePhase_distribution=prePhase_distribution[:,:halfM]

			if(saveDataForAnimation and step%animationDataInterval==0):
				animationDataFile.write(prePhase_distribution.tobytes())

			if step%50 == 0:
				print(step,out[1:5])

			if step%500==0:
				save_path = saver.save(sess, folder+sessionName)	#save sess	

			if step in [0,1,2,4,8,16,32,64,128,256,512] or step%1000==0 or step==current_global_step+iterationToDo:
				#save the distribution of phaseXY as an image
				phaseXY_distribution=phaseXY_distribution[:,:halfM]
				phaseOutput=phaseOutput[:halfM]%(2*np.pi)
				
				util.savePrePhaseAndPhasePlot(prePhase_distribution,phaseXY_distribution,imageFolder+str(step)+'.png',fig)

				#want to draw rainbow loop in parallel
				util.savePhaseRainbowPlot(centerPose,phaseOutput, imageFolder+'r'+str(step)+'.png', fig=figRainbow)


		save_path = saver.save(sess, folder+sessionName)	#save sess

		if(saveDataForAnimation):
			animationDataFile.close()

	else:
		saver.restore(sess, folder+sessionName)
		#observed the learned network by saving phaseRad 
		#as an additional dimension to the training data
		
		#extract phase from test data
		testPhaseOutput,testPrePhaseOutput=sess.run([phaseRad,prePhase],feed_dict={firstLayerPlace:testNowState})
		testPhaseOutput = testPhaseOutput%(2*np.pi)

		print("Phase extraction complete.")
		print("Generating html output...")

		#read phaseGroundTruth
		gtList=pickle.load(open(input+'data_test_phaseGroundTruth.pkl','rb')) 
		gtPhase=gtList[0][regressWing+windowWing:regressWing+windowWing+testCount]

		testOutDict={
			'phaseGroundTruth':gtPhase,
			'phasePredict':testPhaseOutput,
			'centerPose':testCenterPose,
			'wMean':wMean,
			'wSD':wSD
		}
		
		pickle.dump(testOutDict,open(folder+'testOutDict.pkl','wb'))

		import plotly.graph_objs as go

		testDataWithPhase=np.vstack([testCenterPose,testPhaseOutput])
		DwithPhase=D+1

		dataNow=[]
		
		for i in range(0,DwithPhase):
			
			if(i<D):
				legend='Data#'+str(i+1)
			if(i==D):
				legend='Phase (rad)'

			dataNow.append(go.Scatter(
				x=list(range(nowState.shape[1])),
				y=testDataWithPhase[i,:],
				name=legend
			))

		#for plotly 3
		#plotly.offline.plot(dataNow, filename=folder+'phaseAgainstTime_test.html')

		#for plotly 4
		pFig=go.Figure(
			data=dataNow
		)
		pFig.write_html(folder+'phaseAgainstTime_test.html')

		print('Done')
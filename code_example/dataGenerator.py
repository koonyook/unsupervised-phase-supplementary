import random
import numpy as np
from random import randint
#import allProtoClass_pb2 as pb
from copy import deepcopy

def sampleFromCircle(radius,trainsetSize):
	ans=np.zeros([trainsetSize,2])
	for i in range(trainsetSize):
		angle=random.uniform(0,2*np.pi)
		ans[i]=radius*np.cos(angle),radius*np.sin(angle)

	return ans

def sampleFromCircleFlow(radius, rotationRad, trainsetSize):
	ans=np.zeros([trainsetSize,2])
	shiftedAns=np.zeros([trainsetSize,2])
	for i in range(trainsetSize):
		angle=random.uniform(0,2*np.pi)
		ans[i]=radius*np.cos(angle),radius*np.sin(angle)

		angle=angle+rotationRad
		shiftedAns[i]=radius*np.cos(angle),radius*np.sin(angle)

	return ans,shiftedAns

def deformCircleToInf(data):
	ans=np.zeros([len(data),2])
	for i in range(len(data)):
		x,y=data[i]
		ans[i]=x,x*y*2.5

	return ans

def sampleFromInfinityFlow(trainSize):
	beginPoint,endPoint=sampleFromCircleFlow(0.75,10.0*np.pi/180.0,trainSize)
	beginPoint=deformCircleToInf(beginPoint)
	endPoint=deformCircleToInf(endPoint)
	return beginPoint,endPoint


def gen():
	t=np.linspace(0.0, 1.0, num=100)
	t=t*12+3	#change from [0,1] to [3,15]
	return t*0.04*np.sin(t), t*0.04*np.cos(t)

def circleTransform(t):
	return 0.75*np.cos(t*2*np.pi),0.75*np.sin(t*2*np.pi)

def ellipseTransform(t):
	x=np.sin(t*2*np.pi+0.75)
	y=np.cos(t*2*np.pi)
	return 0.75*x,0.75*y

def beanTransform(t):
	x=np.sin(t*2*np.pi+0.75)
	y=np.cos(t*2*np.pi)
	return 0.75*x,0.75*y*y*y

def ghostTransform(t):
	x=np.sin(t*2*np.pi+0.75)+0.3*np.sin(t*4*np.pi+0.5)
	y=np.sin(t*2*np.pi)+0.1*np.sin(t*6*np.pi+0.1)
	return 0.8*x-0.15,0.9*y*y*y

def triangleTransform(t):
	x=[]
	y=[]
	for e in np.fmod(t,1)*3:
		if e<1.0:	#from 
			f=e-0.0
			x.append((-0.4)*(1-f) + 0.7*f)
			y.append((-0.7)*(1-f) + 0.1*f)
		elif e<2.0:
			f=e-1.0
			x.append((0.7)*(1-f) + (-0.6)*f)
			y.append((0.1)*(1-f) + (0.6)*f)
		else:
			f=e-2.0
			x.append((-0.6)*(1-f) + (-0.4)*f)
			y.append((0.6)*(1-f) + (-0.7)*f)

	return x,y


def sTransformOneWay(t):	#old
	x=np.sin(t*2*np.pi)
	y=t-0.5
	return 0.75*x,1.5*y

def sTransform(t):	#go back and forth
	x=[]
	y=[]
	for e in np.fmod(t,1):
		if e<0.5:
			x.append(np.sin(e*4*np.pi))
			y.append(e-0.25)
		else:
			x.append(-np.sin(e*4*np.pi))
			y.append(0.75-e)
	x=np.array(x)
	y=np.array(y)
	return 0.75*x,3*y

def spiralTransform(t):
	t=t*12+3	#change from [0,1] to [3,15]
	return t*0.04*np.sin(t), t*0.04*np.cos(t)

def bowTransform(t):
	t=t*2*np.pi
	return 0.8*np.cos(t), 1.2*np.sin(t)*np.cos(t)

def heartTransform(t):
	t=t*2*np.pi
	return 16*np.sin(t)**3, 13*np.cos(t)-5*np.cos(2*t)-2*np.cos(3*t)-np.cos(4*t)

def acrobatTransform(t):
	t=t*2*np.pi
	return 5*np.cos(t)*(np.sin(t+0.5)+np.cos(3*t+0.5)), 5*np.sin(t)*(np.sin(t+0.5)-np.cos(3*t+0.5))

def regressForSlope(y):	#only odd number
	l=len(y)
	#x is generated (assume equal time interval), zero at the center, meanX=0
	#x is something like [-3,-2,-1,0,1,2,3]
	x=np.array(range(l))-(l-1)/2
	avgY=np.average(y)
	return np.dot(y-avgY,x)/np.dot(x,x)


def getRegressHeadingDirection(x, regressWing=3):	#each row = 1 sequence in one dimension
	regressLength=regressWing*2+1
	x=x.transpose()
	D,W = x.shape
	r=np.zeros([D,W-regressWing*2])
	for d in range(D):
		for i in range(W-regressWing*2):
			r[d,i]=regressForSlope(x[d,i:i+regressLength])
			
	#done regression
	#time to do normalization
	for i in range(W-regressWing*2):
		r[:,i]=r[:,i]/np.linalg.norm(r[:,i])

	return r.transpose()

def getContinuousHeadingDirection(x, regressWing=3, averageWing=2):	#each row = 1 sequence in one dimension
	regressLength=regressWing*2+1
	averageLength=averageWing*2+1
	x=x.transpose()
	D,W = x.shape
	r=np.zeros([D,W-regressWing*2])
	for d in range(D):
		for i in range(W-regressWing*2):
			r[d,i]=regressForSlope(x[d,i:i+regressLength])
			#r[d,i]=x[d,i+3]-x[d,i]
	#done regression
	#time to do normalization
	for i in range(W-regressWing*2):
		r[:,i]=r[:,i]/np.linalg.norm(r[:,i])

	#then window average to smooth it
	ans=np.zeros([D,W-regressWing*2-averageWing*2])
	for d in range(D):
		for i in range(W-regressWing*2-averageWing*2):
			ans[d,i]=np.sum(r[d,i:i+averageLength])/averageLength

	return ans.transpose()

def samplePoseSequence(transformer,sampleLength,stepForward=0.02,stepVariation=0):
	#generate a sequence with random interval based on stepForward with +-40% uniform variation
	interval=np.random.uniform(stepForward*(1-stepVariation),stepForward*(1+stepVariation),sampleLength)
	currentTime=0
	t=np.zeros(sampleLength)
	for i in range(sampleLength):
		currentTime=currentTime+interval[i]
		t[i]=currentTime 
	x0,y0 = transformer(t)	#still in 2D
	return x0,y0

def sampleFromFunction(transformer,trainSize,stepForward=0.02):	#stepForward in range 0-1
	if(callable(transformer)):
		t0=np.random.uniform(0.0,1.0,trainSize)
		t1=t0+stepForward
		t2=t1+stepForward
		x0,y0=transformer(t0)
		x1,y1=transformer(t1)
		x2,y2=transformer(t2)
		return np.vstack([x0,y0]).transpose(), np.vstack([x1,y1]).transpose(), np.vstack([x2,y2]).transpose()
	elif(transformer=='realData'):	#quick hack
		#do not use stepForward
		f=open("koonWristElbow.SampleListProto","rb")
		#f=open("jamesWristElbow.SampleListProto","rb")
		reader = pb.SampleListProto()
		reader.ParseFromString(f.read())
		f.close()

		wx=[]
		wy=[]

		for sample in reader.list:
			#print sample.WristX,sample.WristY,sample.WristZ
			wx.append(sample.WristX)
			wy.append(sample.WristY)

		#apply a filter to smooth the data
		l=len(wx)

		wxs=deepcopy(wx)
		wys=deepcopy(wy)

		wing=2
		for (s,a) in zip([wxs,wys],[wx,wy]):
			for i in range(len(wx)):
				if(i-wing>=0 and i+wing+1<l):
					s[i]=sum(a[i-wing:i+wing+1])/(wing*2+1)

		#crop head and tail
		wxs=wxs[wing:l-wing]
		wys=wys[wing:l-wing]

		#scale the data
		meanX=np.mean(wxs)
		meanY=np.mean(wys)
		wxs=wxs-meanX
		wys=wys-meanY
		stdX=np.std(wxs)
		stdY=np.std(wys)
		wxs=wxs/(2*stdX)*0.9
		wys=wys/(2*stdY)*0.9

		#randomly select data for training
		if(trainSize>len(wxs)-2):
			#create from all of them
			return np.vstack([wxs[0:len(wxs)-2],wys[0:len(wxs)-2]]).transpose(), np.vstack([wxs[1:len(wxs)-1],wys[1:len(wxs)-1]]).transpose(), np.vstack([wxs[2:len(wxs)],wys[2:len(wxs)]]).transpose()
		else:
			#use random
			x0=[]
			y0=[]
			x1=[]
			y1=[]
			x2=[]
			y2=[]
			for i in range(trainSize):
				r=randint(0,len(wxs)-3)
				x0.append(wxs[r])
				y0.append(wys[r])
				x1.append(wxs[r+1])
				y1.append(wys[r+1])
				x2.append(wxs[r+2])
				y2.append(wys[r+2])
			return np.vstack([x0,y0]).transpose(), np.vstack([x1,y1]).transpose(), np.vstack([x2,y2]).transpose()

def sampleFromBean(trainSize,stepForward=0.02):	#stepForward in range 0-1
	t0=np.random.uniform(0.0,1.0,trainSize)
	t1=t0+stepForward
	x0,y0=beanTransform(t0)
	x1,y1=beanTransform(t1)

	return np.vstack([x0,y0]).transpose(), np.vstack([x1,y1]).transpose()

def addGaussianNoise(patches,sd=0.06):
	return patches+np.random.normal(0,sd,size=patches.shape)

def addOrthogonalNoise(previousPatches,currentPatches,nextPatches,minSD=0.04):
	#first, add noise in all direction. Then, project it to a specific hyperplane
	preLength=np.linalg.norm(currentPatches-previousPatches,axis=1)
	postLength=np.linalg.norm(nextPatches-currentPatches,axis=1)
	noiseSize=0.5*np.minimum(preLength,postLength)	#1SD of noise 
												#(if it move fast, noise can be large. if move slow, noise should be small)
												#just don't want the moving path to get overwhelmed by these noise

	noiseSize=np.maximum(noiseSize,minSD)	#noise standard deviation cannot be too small

	noise = np.random.normal(0,1,size=currentPatches.shape)*np.reshape(noiseSize,(-1,1))	#noise in all direction

	#now must project this noise into hyperplanes
	#first, get unit normal vector
	jumpVector=nextPatches-previousPatches
	normalVector=jumpVector/np.reshape(np.linalg.norm(jumpVector,axis=1),(-1,1))

	#second, project noise to that unit normal vector
	toBeRemoved=np.reshape(np.sum(noise*normalVector,axis=1),(-1,1))*normalVector
	projectedNoise=noise-toBeRemoved

	return currentPatches+projectedNoise

def addOrthogonalNoise2(currentPatches,nextPatches,velocityPatches,minSD=0.04):
	#first, add noise in all direction. Then, project it to a specific hyperplane
	#preLength=np.linalg.norm(currentPatches-previousPatches,axis=1)
	postLength=np.linalg.norm(nextPatches-currentPatches,axis=1)
	noiseSize=0.5*postLength
	#noiseSize=0.5*np.minimum(preLength,postLength)	#1SD of noise 
												#(if it move fast, noise can be large. if move slow, noise should be small)
												#just don't want the moving path to get overwhelmed by these noise

	noiseSize=np.maximum(noiseSize,minSD)	#noise standard deviation cannot be too small

	noise = np.random.normal(0,1,size=currentPatches.shape)*np.reshape(noiseSize,(-1,1))	#noise in all direction

	#now must project this noise into hyperplanes
	#first, get unit normal vector
	#jumpVector=nextPatches-previousPatches
	jumpVector=velocityPatches
	normalVector=jumpVector/np.reshape(np.linalg.norm(jumpVector,axis=1),(-1,1))

	#second, project noise to that unit normal vector
	toBeRemoved=np.reshape(np.sum(noise*normalVector,axis=1),(-1,1))*normalVector
	projectedNoise=noise-toBeRemoved

	return currentPatches+projectedNoise

def addVelocityNoise(patches,sdRotation=30,minScale=0.0):
	#print patches.shape
	ans=np.zeros(patches.shape)
	for i in range(patches.shape[0]):
		
		scale = np.random.uniform(minScale,1.0)

		theta = np.random.normal(0.0,sdRotation*np.pi/180)
		rotationMatrix = np.array([	[np.cos(theta), -np.sin(theta)], 
                         				[np.sin(theta),  np.cos(theta)]])

		ans[i,:] = np.matmul(patches[i,:],rotationMatrix)*scale

	return ans

if __name__ == '__main__':
	import matplotlib.pyplot as plt
	
	#fig=plt.figure(figsize=(6, 6))
	#x,y=gen()
	#plt.scatter(x,y)
	#plt.axis('equal')
	#plt.show()
	
	#a=sampleFromCircle(0.75,100)
	#a=deformCircleToInf(a)
	#plt.scatter(a[:,0],a[:,1])
	#plt.show()
	
	#time=np.linspace(0,10,1230)	#0 to 1 is one round
	
	totalRound=150
	samplePerRound=100

	timeShuffled=[]
	for i in range(totalRound*samplePerRound):
		timeShuffled.append(random.uniform(0.0,totalRound))
	
	timeShuffled.sort()
	time=np.array(timeShuffled)
	#x,y=triangleTransform(time)
	#x,y=ghostTransform(time)
	#x,y=bowTransform(time)
	#x,y=sTransform(time)
	#x,y=heartTransform(time)
	x,y=acrobatTransform(time)

	#save to .dat
	dataToSave=np.vstack([x,y])
	dataToSave.dump('s.dat')

	index=np.array(list(range(totalRound*samplePerRound)))
	plt.plot(index,x,'.r-')
	plt.plot(index,y,'.b-')
	plt.show()

	plt.cla()
	plt.scatter(x,y)
	plt.axis('equal')
	plt.show()
	

	#addVelocityNoise(np.transpose(np.vstack([x,y])))

	#plt.plot(time,y)
	#plt.show()
	
	'''
	#a,b=sampleFromCircleFlow(0.75,10.0*np.pi/180.0, 100)
	#a,b=sampleFromInfinityFlow(100)
	#a,b=sampleFromBean(100)
	a,b=sampleFromFunction(ellipsTransform,100)
	#print(len(a))
	plt.axis('equal')	#allow true square scaling
	v=b-a
	plt.quiver(a[:,0], a[:,1], v[:,0], v[:,1], scale=1, units='x')
	plt.axis([-1.1,1.1,-1.1,1.1])
	plt.show()
	'''
	
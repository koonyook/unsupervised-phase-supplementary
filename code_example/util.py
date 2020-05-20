import random
import math
import numpy as np
from random import randint
from copy import deepcopy
import os
import matplotlib
if os.name != 'nt':
	matplotlib.use('Agg') 
import matplotlib.pyplot as plt

def expandScale(scale,sphereCount):
	ans=np.zeros([scale.shape[0]+sphereCount*2,1])
	for s in range(sphereCount):
		ans[s*3+0,0]=scale[s]
		ans[s*3+1,0]=scale[s]
		ans[s*3+2,0]=scale[s]
	
	ans[sphereCount*3:,0]=scale[sphereCount:,0]

	return ans

def projectToPlane(planeNormal:np.ndarray,vector:np.ndarray,keepSize=False):
	unitPlaneNormal=planeNormal/np.linalg.norm(planeNormal,axis=0,keepdims=True)	#normalize to a unit vector
	toBeRemoved = unitPlaneNormal*(np.sum(unitPlaneNormal*vector,axis=0,keepdims=True))
	projection=vector-toBeRemoved
	if keepSize:
		originalSize=np.linalg.norm(vector,axis=0,keepdims=True)
		return projection/np.linalg.norm(projection,axis=0,keepdims=True)*originalSize	#more like rotation to the plane
	else:
		return projection #simple projection to the plane


def regressForSlope(y):	#only odd number
	l=len(y)
	#x is generated (assume equal time interval), zero at the center, meanX=0
	#x is something like [-3,-2,-1,0,1,2,3]
	x=np.array(range(l))-(l-1)/2
	avgY=np.average(y)
	return np.dot(y-avgY,x)/np.dot(x,x)

#follow Andrew's convention
def getRegressHeadingDirectionWithSphere(x, sphereCount, regressWing=3):	#each row = 1 sequence in one dimension
	regressLength=regressWing*2+1
	D,W = x.shape
	r=np.zeros([D,W-regressWing*2])
	#regress each individual dimension
	for d in range(D):
		for i in range(W-regressWing*2):
			r[d,i]=regressForSlope(x[d,i:i+regressLength])
			
	#correct sphere dimensions by rotate them to be orthogonal to the current radius
	#these are not unit sphere
	for s in range(sphereCount):
		for i in range(W-regressWing*2):
			midPoint=x[s*3:s*3+3,[i+regressWing]]
			heading=r[s*3:s*3+3,[i]]
			r[s*3:s*3+3,[i]]=projectToPlane(midPoint,heading,keepSize=True)
	
	#normalization
	for i in range(W-regressWing*2):
		r[:,i]=r[:,i]/np.linalg.norm(r[:,i])

	return r

#follow Andrew's convention, not tested
def addOrthogonalNoiseNDWithSphere(currentPatches,nextPatches,velocityPatches,sphereCount,radius,minSD=0.04,repeat=1):
	#first, add noise in all direction. Then, project it to a specific hyperplane
	#preLength=np.linalg.norm(currentPatches-previousPatches,axis=1)
	postLength=np.linalg.norm(nextPatches-currentPatches,axis=0,keepdims=True)
	noiseSize=postLength/2	
	#noiseSize=0.5*np.minimum(preLength,postLength)	#1SD of noise 
												#(if it move fast, noise can be large. if move slow, noise should be small)
												#just don't want the moving path to get overwhelmed by these noise

	noiseSize=np.maximum(noiseSize,minSD)	#noise standard deviation cannot be too small


	noisyPoseList=[]
	noiseDirectionList=[]
	noiseMagnitudeList=[]

	print('generating orthogonal noise...')
	for i in range(repeat):
		#print('round '+str(i)+'/'+str(repeat))
		#noise = np.random.normal(0,1,size=currentPatches.shape)*noiseSize	#noise in all direction
		noise = np.random.randn(currentPatches.shape[0],currentPatches.shape[1])*noiseSize	#noise in all direction

		#now must project this noise into hyperplanes
		#first, get unit normal vector
		#jumpVector=nextPatches-previousPatches
		jumpVector=velocityPatches
		normalVector=jumpVector/np.linalg.norm(jumpVector,axis=0,keepdims=True)

		#second, project noise to that unit normal vector
		toBeRemoved=np.sum(noise*normalVector,axis=0,keepdims=True)*normalVector
		projectedNoise=noise-toBeRemoved	

		#the projectedNoise need spherical correction (currentPatches+projectedNoise must be on sphere)
		noisyPose=currentPatches+projectedNoise
		surfaceDistance=np.zeros([sphereCount,currentPatches.shape[1]])
		for s in range(sphereCount):
			noisyPose[s*3:s*3+3,:]*=radius[s]/np.linalg.norm(noisyPose[s*3:s*3+3,:],axis=0)	#project them to sphere surface

			
			#noiseMagnitude must calculate with distance on a sphere surface as one dimension
			
			#first, find spherical surface distance from currentPatches to noisyPose
			unitNoisyPose=noisyPose[s*3:s*3+3,:]/radius[s]
			unitOriginal=currentPatches[s*3:s*3+3,:]/radius[s]
			dotProduct=np.sum(unitNoisyPose*unitOriginal,axis=0,keepdims=True)
			angle=np.arccos(np.minimum(1,np.maximum(-1,dotProduct)))	#distance in a unit sphere
			surfaceDistance[s,:]=radius[s]*angle

		otherDistance=projectedNoise[sphereCount*3:,:]

		mixedDistance=np.vstack([surfaceDistance,otherDistance])	#one row for one sphere (surface distance, positive only), one row for one scalar dimension (can be negative) 
		noiseMagnitude=np.linalg.norm(mixedDistance,axis=0,keepdims=True)
		unitMixedDistance=mixedDistance/np.linalg.norm(mixedDistance,axis=0,keepdims=True)

		#noiseDirection must be tangent to the sphere, at the noisy spot, and point away from the original point, and has unit norm
		roughNoiseDirection=noisyPose-currentPatches
		noiseDirection=np.zeros(noisyPose.shape)
		for s in range(sphereCount):
			tangentDirection=projectToPlane(currentPatches[s*3:s*3+3,:],roughNoiseDirection[s*3:s*3+3,:],keepSize=False)	#just want direction
			tangentDirection/=np.linalg.norm(tangentDirection,axis=0,keepdims=True)
			noiseDirection[s*3:s*3+3,:]=tangentDirection*unitMixedDistance[[s],:]
		noiseDirection[sphereCount*3:,:]=unitMixedDistance[sphereCount:,:]

		noisyPoseList.append(noisyPose)			  #done (not tested yet)
		noiseDirectionList.append(noiseDirection) #done (not tested yet)
		noiseMagnitudeList.append(noiseMagnitude) #done (not tested yet)

	return np.hstack(noisyPoseList), np.hstack(noiseDirectionList), np.hstack(noiseMagnitudeList)
	#return currentPatches+projectedNoise, noiseDirection, noiseMagnitude

#follow Andrew's convention, not tested
def addHeadNoiseNDWithSphere(nowPose,nowHead,sphereCount,noiseSD=0.5,repeat=1):	#sd=0.5 is about 30 degrees deviation
	
	noisyHeadList=[]
	noiseDirectionList=[]
	noiseMagnitudeList=[]
	for i in range(repeat):
		#in this high dimensional case, I add noise to the unit direction, and project it back to a hypersphere using normalization
		noise = np.random.randn(nowHead.shape[0],nowHead.shape[1])*noiseSD
		tmp = nowHead+noise 	#add random noise in all direction
		noisyPatches = tmp/np.linalg.norm(tmp,axis=0,keepdims=True)	#projected back to the unit hypersphere

		#fix heading direction of spherical dimension (must be tangent to sphere surface)
		for s in range(sphereCount):
			noisyPatches[s*3:s*3+3,:]=projectToPlane(nowPose[s*3:s*3+3,:],noisyPatches[s*3:s*3+3,:],keepSize=True)


		#direction of noise in heading direction must be orthogonal to the noisy unit vector
		nonOrthoVec = noisyPatches-nowHead
		#project nonOrthoVec to the direction of noisyPatches
		toBeRemoved = np.sum(nonOrthoVec*noisyPatches,axis=0,keepdims=True)*noisyPatches #a short vector in noisyPatches direction
		orthoVec = nonOrthoVec-toBeRemoved

		noiseDirection = orthoVec/np.linalg.norm(orthoVec,axis=0,keepdims=True)

		noisyHeadList.append(noisyPatches)
		noiseDirectionList.append(noiseDirection)
		noiseMagnitudeList.append(np.linalg.norm(nonOrthoVec,axis=0,keepdims=True))

	return np.hstack(noisyHeadList), np.hstack(noiseDirectionList), np.hstack(noiseMagnitudeList) 

def randomOuterPoseWithSphere(sphereCount,radius,scalarRange,count):
	#scalarRange does not contain sphere-related dimension

	outerPose=np.zeros([sphereCount*3+scalarRange.shape[0],count])

	#random possible pose
	#spherical data need Gaussian random and normalize to sphere
	for s in range(sphereCount):
		r=np.random.randn(3,count)
		r=r/np.linalg.norm(r,axis=0,keepdims=True)*radius[s]
		outerPose[s*3:s*3+3,:]=r
	
	#scalar dimension need uniform random
	outerPose[sphereCount*3:,:]=np.random.rand(scalarRange.shape[0],count)*(scalarRange[:,[1]]-scalarRange[:,[0]])+scalarRange[:,[0]]

	return outerPose

def searchForNearestNeighborAndDistance(q,nowPose,sphereCount,radius):
	count=q.shape[1]
	#search for the nearest neighbor
	nearest=np.zeros([nowPose.shape[0],count])
	distance=np.zeros([1,count])
	for i in range(count):
		p=q[:,[i]]
		sumSquareDistance=np.zeros([1,nowPose.shape[1]])	#to compare
		for s in range(sphereCount):
			dot=np.sum(p[s*3:s*3+3,:]*nowPose[s*3:s*3+3,:],axis=0,keepdims=True)/(radius[s]*radius[s])
			angle=np.arccos(np.maximum(-1,np.minimum(1,dot)))
			sumSquareDistance+=np.square(angle*radius[s])
		sumSquareDistance+=np.sum(np.square(p[sphereCount*3:,:]-nowPose[sphereCount*3:,:]),axis=0,keepdims=True)
	
		#hypothesis: with 1 nearest neighbor, it might be too sensive to outliers (average of 10 nearest neighbor might be better)
		#however, if the outliers is not too far, the dense denoising strategy nearby the collected data will do the job
		minIndex=np.argmin(sumSquareDistance)
		nearest[:,[i]]=nowPose[:,[minIndex]]
		distance[0,i]=np.sqrt(sumSquareDistance[0,minIndex])
	
	return nearest,distance

def randomOuterPoseAndExpectedGradientWithSphere(nowPose,sphereCount,radius,scalarRange,count,kNearest=25):
	#scalarRange does not contain sphere-related dimension

	outerPose=np.zeros([nowPose.shape[0],count])

	#random possible pose
	#spherical data need Gaussian random and normalize to sphere
	for s in range(sphereCount):
		r=np.random.randn(3,count)
		r=r/np.linalg.norm(r,axis=0,keepdims=True)*radius[s]
		outerPose[s*3:s*3+3,:]=r
	
	#scalar dimension need uniform random
	outerPose[sphereCount*3:,:]=np.random.rand(scalarRange.shape[0],count)*(scalarRange[:,[1]]-scalarRange[:,[0]])+scalarRange[:,[0]]

	#search for the nearest neighbor
	target=np.zeros([nowPose.shape[0],count])
	for i in range(count):
		p=outerPose[:,[i]]
		sumSquareDistance=np.zeros([1,nowPose.shape[1]])	#to compare
		for s in range(sphereCount):
			dot=np.sum(p[s*3:s*3+3,:]*nowPose[s*3:s*3+3,:],axis=0,keepdims=True)/(radius[s]*radius[s])
			angle=np.arccos(np.maximum(-1,np.minimum(1,dot)))
			sumSquareDistance+=np.square(angle*radius[s])
		sumSquareDistance+=np.sum(np.square(p[sphereCount*3:,:]-nowPose[sphereCount*3:,:]),axis=0,keepdims=True)
	
		#hypothesis: with 1 nearest neighbor, it might be too sensive to outliers (average of 20 nearest neighbor might be better)
		#however, if the outliers is not too far, the dense denoising strategy nearby the collected data will do the job
		if kNearest<=1:
			target[:,[i]]=nowPose[:,[np.argmin(sumSquareDistance)]]	#one nearest neighbor is not good enough
		else:
			kNearestIndex=np.argpartition(sumSquareDistance.flatten(),kNearest)[0:kNearest]
			kNearestPose=nowPose[:,kNearestIndex]
			#get average from kNearestPose (and must stay inside the possible space)
			targetTemp=np.mean(kNearestPose,axis=1,keepdims=True)
			for s in range(sphereCount):
				targetTemp[s*3:s*3+3,:]=radius[s]*targetTemp[s*3:s*3+3,:]/np.linalg.norm(targetTemp[s*3:s*3+3,:])

			target[:,[i]]=targetTemp

	#calculate expected gradient (point away from the nearest neighbor)
	# outerPose -> target
	surfaceDistance=np.zeros([sphereCount,count])
	for s in range(sphereCount):
		dot=np.sum(outerPose[s*3:s*3+3,:]*target[s*3:s*3+3,:],axis=0,keepdims=True)/(radius[s]*radius[s])
		angle=np.arccos(np.maximum(-1,np.minimum(1,dot)))
		surfaceDistance[s,:]=angle*radius[s]

	otherDistance=outerPose[sphereCount*3:,:]-target[sphereCount*3:,:]		#critical bug fixed
	
	mixedDistance = np.vstack([surfaceDistance,otherDistance])
	unitMixedDistance = mixedDistance/np.linalg.norm(mixedDistance,axis=0,keepdims=True)

	#noiseDirection must be tangent to the sphere, at the noisy spot, and point away from the original point, and has unit norm
	roughDirection=outerPose-target
	expectedGradient=np.zeros(outerPose.shape)
	for s in range(sphereCount):
		tangentDirection=projectToPlane(outerPose[s*3:s*3+3,:],roughDirection[s*3:s*3+3,:],keepSize=False)	#just want direction
		tangentDirection/=np.linalg.norm(tangentDirection,axis=0,keepdims=True)
		expectedGradient[s*3:s*3+3,:]=tangentDirection*unitMixedDistance[[s],:]
	expectedGradient[sphereCount*3:,:]=unitMixedDistance[sphereCount:,:]

	return outerPose,expectedGradient

def randomOuterPoseAndExpectedGradientWithSphere_farAndNear(nowPose,sphereCount,radius,scalarRange,count,kNearest=25,includeNearSide=True,nearRatio=1.0):
	#nearRatio=1.0 is adding point right at the target
	#scalarRange does not contain sphere-related dimension
	print('preparing outerPose data...')
	outerPose=np.zeros([nowPose.shape[0],count])

	#random possible pose
	#spherical data need Gaussian random and normalize to sphere
	print('random points...')
	for s in range(sphereCount):
		r=np.random.randn(3,count)
		r=r/np.linalg.norm(r,axis=0,keepdims=True)*radius[s]
		outerPose[s*3:s*3+3,:]=r
	
	#scalar dimension need uniform random
	outerPose[sphereCount*3:,:]=np.random.rand(scalarRange.shape[0],count)*(scalarRange[:,[1]]-scalarRange[:,[0]])+scalarRange[:,[0]]

	#search for the nearest neighbor
	print('nearest neighbor search...')
	target=np.zeros([nowPose.shape[0],count])
	beforeTarget=np.zeros(target.shape)
	for i in range(count):
		print(str(i)+'/'+str(count),end='\r')
		p=outerPose[:,[i]]
		sumSquareDistance=np.zeros([1,nowPose.shape[1]])	#to compare
		for s in range(sphereCount):
			dot=np.sum(p[s*3:s*3+3,:]*nowPose[s*3:s*3+3,:],axis=0,keepdims=True)/(radius[s]*radius[s])
			angle=np.arccos(np.maximum(-1,np.minimum(1,dot)))
			sumSquareDistance+=np.square(angle*radius[s])
		sumSquareDistance+=np.sum(np.square(p[sphereCount*3:,:]-nowPose[sphereCount*3:,:]),axis=0,keepdims=True)
	
		#hypothesis: with 1 nearest neighbor, it might be too sensive to outliers (average of 20 nearest neighbor might be better)
		#however, if the outliers is not too far, the dense denoising strategy nearby the collected data will do the job
		if kNearest<=1:
			target[:,[i]]=nowPose[:,[np.argmin(sumSquareDistance)]]	#one nearest neighbor is not good enough
		else:
			kNearestIndex=np.argpartition(sumSquareDistance.flatten(),kNearest)[0:kNearest]
			kNearestPose=nowPose[:,kNearestIndex]
			#get average from kNearestPose (and must stay inside the possible space)
			targetTemp=np.mean(kNearestPose,axis=1,keepdims=True)
			for s in range(sphereCount):
				targetTemp[s*3:s*3+3,:]=radius[s]*targetTemp[s*3:s*3+3,:]/np.linalg.norm(targetTemp[s*3:s*3+3,:])

			target[:,[i]]=targetTemp

	#calculate expected gradient (point away from the nearest neighbor)
	print('calculate expected gradient...')
	# outerPose -> target
	surfaceDistance=np.zeros([sphereCount,count])
	for s in range(sphereCount):
		dot=np.sum(outerPose[s*3:s*3+3,:]*target[s*3:s*3+3,:],axis=0,keepdims=True)/(radius[s]*radius[s])
		angle=np.arccos(np.maximum(-1,np.minimum(1,dot)))
		surfaceDistance[s,:]=angle*radius[s]

	otherDistance=outerPose[sphereCount*3:,:]-target[sphereCount*3:,:]		#critical bug fixed
	
	mixedDistance = np.vstack([surfaceDistance,otherDistance])
	unitMixedDistance = mixedDistance/np.linalg.norm(mixedDistance,axis=0,keepdims=True)

	#noiseDirection must be tangent to the sphere, at the noisy spot, and point away from the original point, and has unit norm
	roughDirection=outerPose-target

	#on the remote side
	expectedGradient=np.zeros(outerPose.shape)
	for s in range(sphereCount):
		tangentDirection=projectToPlane(outerPose[s*3:s*3+3,:],roughDirection[s*3:s*3+3,:],keepSize=False)	#just want direction
		tangentDirection/=np.linalg.norm(tangentDirection,axis=0,keepdims=True)
		expectedGradient[s*3:s*3+3,:]=tangentDirection*unitMixedDistance[[s],:]
	expectedGradient[sphereCount*3:,:]=unitMixedDistance[sphereCount:,:]

	if includeNearSide:
		'''
		#can reuse roughDirection but project differently
		expectedGradientNear=np.zeros(outerPose.shape)
		for s in range(sphereCount):
			tangentDirection=projectToPlane(target[s*3:s*3+3,:],roughDirection[s*3:s*3+3,:],keepSize=False)
			tangentDirection/=np.linalg.norm(tangentDirection,axis=0,keepdims=True)
			expectedGradientNear[s*3:s*3+3,:]=tangentDirection*unitMixedDistance[[s],:]
		expectedGradientNear[sphereCount*3:,:]=unitMixedDistance[sphereCount:,:]

		return np.hstack([outerPose,target]),np.hstack([expectedGradient,expectedGradientNear])
		'''

		#use spherical interpolation between outerPose and target to get "beforeTarget point" 		
		for i in range(count):
			for s in range(sphereCount):
				#need rotation axis
				fullAngle=surfaceDistance[s,i]/radius[s]
				axisOfRotation=np.cross(outerPose[s*3:s*3+3,i],target[s*3:s*3+3,i])
				axisOfRotation/=np.linalg.norm(axisOfRotation)	
				#rotate currentPoint with axis and angle
				beforeTarget[s*3:s*3+3,i]=rotateAxisAngle(outerPose[s*3:s*3+3,i],axisOfRotation,fullAngle*nearRatio)
				#adjust it back to the correct radius (remove collective error)
				beforeTarget[s*3:s*3+3,i]=radius[s]*beforeTarget[s*3:s*3+3,i]/np.linalg.norm(beforeTarget[s*3:s*3+3,i])

		beforeTarget[sphereCount*3:,:]=nearRatio*target[sphereCount*3:,:]+(1-nearRatio)*outerPose[sphereCount*3:,:]	#linear interpolation for scalar value

		#they are equal
		#print(target)
		#print("\n==========\n")
		#print(beforeTarget)

		expectedGradientNear=np.zeros(outerPose.shape)
		for s in range(sphereCount):
			tangentDirection=projectToPlane(beforeTarget[s*3:s*3+3,:],roughDirection[s*3:s*3+3,:],keepSize=False)
			tangentDirection/=np.linalg.norm(tangentDirection,axis=0,keepdims=True)
			expectedGradientNear[s*3:s*3+3,:]=tangentDirection*unitMixedDistance[[s],:]

		expectedGradientNear[sphereCount*3:,:]=unitMixedDistance[sphereCount:,:]

		return np.hstack([outerPose,beforeTarget]),np.hstack([expectedGradient,expectedGradientNear])

	else:
		return outerPose,expectedGradient

def rotation_matrix(axis, angle):
	"""
	Return the rotation matrix associated with counterclockwise rotation about
	the given axis by theta radians.
	https://stackoverflow.com/questions/6802577/python-rotation-of-3d-vector
	"""
	axis = np.asarray(axis)
	axis = axis/math.sqrt(np.dot(axis, axis))
	a = math.cos(angle/2.0)
	b, c, d = -axis*math.sin(angle/2.0)
	aa, bb, cc, dd = a*a, b*b, c*c, d*d
	bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
	return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
					 [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
					 [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])

#v = [3, 5, 0]
#axis = [4, 4, 1]
#theta = 1.2 

def rotateAxisAngle(vector, axis, angle):
	return np.dot(rotation_matrix(axis,angle), vector)

def easyMoveOnSphere(currentPoint,moveVector,sphereCount,scale,scalarBound=None):	#for one point
	#simply add and normalize
	currentPoint+=moveVector
	for s in range(sphereCount):
		currentPoint[s*3:s*3+3,:]=currentPoint[s*3:s*3+3,:]*scale[s]/np.linalg.norm(currentPoint[s*3:s*3+3,:])

	if scalarBound is not None:
		currentPoint[sphereCount*3:,:]=np.maximum(scalarBound[:,[0]],np.minimum(scalarBound[:,[1]],currentPoint[sphereCount*3:,:]))

	return currentPoint

def preciseMoveOnSphere(currentPoint,moveVector,sphereCount,radius,scalarBound=None):	#for one point
	#move on sphere
	for s in range(sphereCount):
		surfaceDistance=np.linalg.norm(moveVector[s*3:s*3+3,:])
		angle=surfaceDistance/radius[s]
		axisOfRotation=np.cross(currentPoint[s*3:s*3+3,0],moveVector[s*3:s*3+3,0])
		axisOfRotation/=np.linalg.norm(axisOfRotation)	
		#rotate currentPoint with axis and angle
		currentPoint[s*3:s*3+3,0]=rotateAxisAngle(currentPoint[s*3:s*3+3,0],axisOfRotation,angle)
		#adjust it back to the correct radius (remove collective error)
		currentPoint[s*3:s*3+3,:]=radius[s]*currentPoint[s*3:s*3+3,:]/np.linalg.norm(currentPoint[s*3:s*3+3,:])

	#move scalar normally
	currentPoint[sphereCount*3:,:]+=moveVector[sphereCount*3:,:]
	if scalarBound is not None:
		currentPoint[sphereCount*3:,:]=np.maximum(scalarBound[:,[0]],np.minimum(scalarBound[:,[1]],currentPoint[sphereCount*3:,:]))

	return currentPoint

def isInside(currentScalar,scalarRange):
	for i in range(currentScalar.shape[0]):
		if currentScalar[i,0]<scalarRange[i,0] or currentScalar[i,0]>scalarRange[i,1]:
			return False
	
	return True

def correctPose(currentPose, sphereCount, radius, scalarBound=None):
	for s in range(sphereCount):
		#adjust it back to the correct radius (remove collective error)
		currentPose[s*3:s*3+3,:]=radius[s]*currentPose[s*3:s*3+3,:]/np.linalg.norm(currentPose[s*3:s*3+3,:])
	
	if scalarBound is not None:
		currentPose[sphereCount*3:,:]=np.maximum(scalarBound[:,[0]],np.minimum(scalarBound[:,[1]],currentPose[sphereCount*3:,:]))

	return currentPose

def correctHead(currentHead, currentPose, sphereCount ):
	#fix heading direction of spherical dimension (must be tangent to sphere surface)
	for s in range(sphereCount):
		currentHead[s*3:s*3+3,:]=projectToPlane(currentPose[s*3:s*3+3,:],currentHead[s*3:s*3+3,:],keepSize=True)
	
	currentHead = currentHead/np.linalg.norm(currentHead,axis=0,keepdims=True)	#projected back to the unit hypersphere

	return currentHead

def saveScatterPlot(XY, filename, fig=None, bound=True):
	if(fig==None):
		fig=plt.figure(figsize=(8, 8))	#in inch

	plt.scatter(XY[0,:],XY[1,:],color='r',s=1)
	plt.axis('equal')	#allow true square scaling
	#plt.axis([-1.5,1.5,-1.5,1.5])

	if bound:
		plt.axis([-1.1,1.1,-1.1,1.1])
		#plt.axis([-1.5,1.5,-1.5,1.5])

	plt.savefig(filename,bbox_inches = 'tight',pad_inches = 0)
	fig.clf()

def saveScatterPlotMovementProfile(XY, filename, fig=None):
	if(fig==None):
		fig=plt.figure()#figsize=(8, 8))	#in inch

	plt.scatter(XY[0,:],XY[1,:],color='r',s=0.2)
	#plt.axis('equal')	#allow true square scaling
	#plt.axis([-1.5,1.5,-1.5,1.5])

	#if bound:
	#	plt.axis([-1.1,1.1,-1.1,1.1])
	#	#plt.axis([-1.5,1.5,-1.5,1.5])

	plt.savefig(filename,bbox_inches = 'tight',pad_inches = 0)
	fig.clf()

def saveScatterPlotWithColors(XY, colors, filename, fig=None, bound=True):
	if(fig==None):
		fig=plt.figure(figsize=(8, 8))	#in inch

	plt.rcParams['image.cmap'] = 'gist_rainbow'
	plt.scatter(XY[0,:],XY[1,:],c=colors,s=0.2,vmin=-np.pi,vmax=np.pi)
	plt.axis('equal')	#allow true square scaling
	plt.colorbar()
	#plt.axis([-1.5,1.5,-1.5,1.5])

	if bound:
		plt.axis([-1.1,1.1,-1.1,1.1])
		#plt.axis([-1.5,1.5,-1.5,1.5])

	plt.savefig(filename,bbox_inches = 'tight',pad_inches = 0)
	fig.clf()

def savePrePhaseAndPhasePlot(prePhaseXY, phaseXY, filename, fig=None):
	if(fig==None):
		fig=plt.figure(figsize=(8, 8))	#in inch
	
	plt.figure(fig.number)

	plt.scatter(phaseXY[0,:],phaseXY[1,:],color='g',s=1)
	plt.scatter(prePhaseXY[0,:],prePhaseXY[1,:],color='r',s=1)
	plt.scatter([0],[0],color='b',s=5)
	plt.axis('equal')	#allow true square scaling

	#plt.axis([-1.1,1.1,-1.1,1.1])

	plt.savefig(filename,bbox_inches = 'tight',pad_inches = 0)
	fig.clf()

def savePhaseRainbowPlot(centerPose,phase, filename, fig=None):
	if(fig==None):
		fig=plt.figure(figsize=(8, 8))	#in inch

	plt.figure(fig.number)

	plt.rcParams['image.cmap'] = 'gist_rainbow'
	plt.scatter(centerPose[0,:],centerPose[1,:],c=phase/np.pi,s=5,vmin=0,vmax=2,zorder=1)
	cb=plt.colorbar(format=matplotlib.ticker.FormatStrFormatter('%g $\pi$'))
	
	plt.axis('equal')	#allow true square scaling
	plt.grid()

	if filename!='':
		plt.savefig(filename,bbox_inches = 'tight',pad_inches = 0)
		fig.clf()
	else:
		plt.show()	

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector,axis=0,keepdims=True)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.sum(v1_u*v2_u,axis=0,keepdims=True), -1.0, 1.0))
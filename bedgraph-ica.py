#!/usr/bin/python

from os.path import splitext
import numpy as np
import sys
from sklearn.decomposition import FastICA
from sklearn.utils.linear_assignment_ import linear_assignment as hungarian
import pywt
import argparse

myColors = ("#85BEFF", "#986300", "#009863", "#F2EC00", "#F23600", "#C21BFF", "#85FFC7")
colorDict = {frozenset([0]):myColors[0], frozenset([1]):myColors[1], frozenset([2]):myColors[2], frozenset([2,1]):myColors[3], frozenset([2,0]):myColors[4], frozenset([1,0]):myColors[5], frozenset([2,1,0]):myColors[6]}
zero = 0.0004

def main():
	parser = argparse.ArgumentParser(description="Performs signal separation on bedgraph files.")
	parser.add_argument("-s", "--signal", dest="S", metavar='FILE', nargs="+", help="Bedgraph signal files", required=True)
	parser.add_argument("-n","--noise", dest="N", metavar='FILE', nargs="*", help="Control signals", required=True)
	parser.add_argument("-b","--bins", dest="smoothBins", metavar="INT", default=20, help="Number of bins to use for hanning window smoothing", type=int)
	parser.add_argument("-C", metavar="INT", help="Number of components for separation", type=int)
	parser.add_argument("-t","--thresh", dest="thresh", metavar="INT", default=80, help="Lower percent of variation from Haar to remove", type=int)
	parser.add_argument('-w','--write', action='store_true', help="Write the output files")
	parser.add_argument('-m','--merge', default=6, help="Merge regions < [6] windows apart", type=int, metavar='INT')
	parser.add_argument('-p','--power', default=5.0, help="Power transform", type=float, metavar='FLOAT')
	parser.add_argument('--boot', default=1, help="Bootstrap ICA (Default: %(default))", type=int, metavar='INT')
	parser.add_argument('-f','--fun',default="cube", help="FastICA function (Default: %(default))", type=str, metavar="STR")
	args = parser.parse_args()
	locations = []
	vals = []
	print("Time Points:\n\t"+'\n\t'.join(args.S)+"\n")
	processFiles(locations, vals, args.S)
	sigI = range(len(args.S)) #signal indices
	if args.N:
		print("Controls:\n\t"+'\n\t'.join(args.N)+"\n")
		processFiles(locations, vals, args.N)
		noiseI = range(len(args.S),len(args.S)+len(args.N)) #noise indices
	vals = np.array(vals)
	print "Normalizing data with size factors"
	normVals = normalize(vals)
	del vals
	if args.C:
		nComps = args.C
	else:
		nComps = len(sigI)
		if noiseI:
			nComps += len(noiseI)
	# Run ICA
	print "Running ICA for %i components" % (nComps,)
	#smoothedVals = smoothData(normVals**(1/5.0), windowSize=args.smoothBins, p=args.thresh)
	smoothedVals = normVals**(1.0/args.power)
	if args.boot > 1:
		countSig = np.zeros(smoothedVals.shape,dtype=np.uint16)
	#stepSize = 5000
	#dataSize = smoothedVals.shape[1]
	#print "Chrom Size: %i"%(dataSize)
	for i in xrange(args.boot):
		#print "Pass: %i"%(i)
		#for sLoc in range(0, dataSize, 100):
		#sEnd = min(sLoc+stepSize, dataSize)
		ica = FastICA(max_iter=300, tol=0.0001, n_components=nComps, fun=args.fun, algorithm="deflation")
		#ica = FastICA(max_iter=300, tol=0.0001, n_components=nComps, fun=args.fun)
		signals = ica.fit_transform(smoothedVals.T)
		#signals = ica.fit_transform(smoothedVals.T[sLoc:sEnd,:])
		print np.around(ica.mixing_, decimals=2)
		#### Pull values ###################################
#		absMixing = np.abs(ica.mixing_)
#		pairs = hungarian(-1*absMixing)
#		print pairs
#		noiseV = []
#		for index in noiseI:
#			noiseV.append(pullVal(ica.mixing_, index, pairs))
#			# switched to hungarian algorithm
#			# http://itakura.kes.tul.cz/zbynek/pubs/SPL.pdf
#		sigV = []
#		for index in sigI:
#			sigV.append(pullVal(ica.mixing_, index, pairs))
		mixing = ica.mixing_
		absMixing = np.absolute(mixing)
		print hungarian(-1.0*absMixing)
		noiseV = []
		for index in noiseI:
			# modify this to make the signal with all same sign and max for noise position noise
			pullVal(absMixing, mixing, index, noiseV)
		sigV = []
		for index in sigI:
			# modify this to use hungarian, but without the noise vector
			pullVal(absMixing, mixing, index, sigV)
		####################################################
		if args.boot > 1:
			updateCount(countSig, signals, sigI, sigV, args.smoothBins, args.thresh)
			#updateCount(countSig[:,sLoc:sEnd], signals, sigI, sigV, args.smoothBins, args.thresh)
	if args.boot > 1:
		writeBoot(countSig, sigI, args.S, locations, args.boot, args.thresh, args.smoothBins)
		return
	if args.write and args.boot == 1: # write output
		print "Writing:"
		for locI,mixI in zip(sigI,sigV):
			if mixI < 0:
				writeSignals(-signals[:,np.abs(mixI)],locations[locI],args.S[locI],locI,args.smoothBins,args.thresh,args.merge)
			else:
				writeSignals(signals[:,mixI],locations[locI],args.S[locI],locI,args.smoothBins,args.thresh, args.merge)

def writeBoot(countM, sigI, inFiles, locations, N, p, bins, pMin=0.80):
	fN = np.float(N)
	maskM = np.zeros((len(sigI), countM.shape[1]), dtype=np.bool)
	for i in sigI:
		baseName = splitext(inFiles[i])[0]
		outFile = "boot_%s.bedgraph"%(baseName)
		outGFF = "boot_%s.gff3"%(baseName)
		OF = open(outFile,'w')
		#smoothCount = haarVec(hannVec(countM[i,:], windowSize=bins), p=p)
		divArray = countM[i]/fN
		for j in xrange(countM.shape[1]):
			outStr = '%s\t%s\t%s\t%.6f\n' % tuple(locations[i][j]+[divArray[j]])
			#outStr = '%s\t%s\t%s\t%.6f\n' % tuple(locations[i][j]+[smoothCount[j]/fN])
			OF.write(outStr)
		OF.close()
		bMask = np.zeros(countM.shape[1], dtype=np.bool)
		bMask[divArray > pMin] = 1 # pMin is the fraction of trials a region has to be true
		#bMask[smoothCount > N*pMin] = 1 # pMin is the fraction of trials a region has to be true
		maskM[i,:]=bMask[:] # update big mask for comparisons
		rBounds = calcRegionBounds(bMask)
		OG = open(outGFF,'w')
		OG.write('##gff-version 3\n#track name="ICA %sbp S%i" gffTags=on\n' % (locations[0][0][2], i+1))
		count = 1
		for s,e in rBounds:
			OG.write("%s\t.\tgene\t%i\t%s\t.\t.\t.\tID=gene%04d;color=%s;\n" % (locations[i][s][0], int(locations[i][s][1])+1, locations[i][e][2], count, myColors[i]))
			count += 1
		OG.close()
	OS = open('segmentation.gff3','w')
	OS.write('##gff-version 3\n#track name="Segmentation %sbp" gffTags=on\n'%(locations[0][0][2]))
	setSigI = set(sigI)
	count = 1
	for s in powerSet(sigI):
		setS = set(s)
		tA = np.ones(countM.shape[1], dtype=np.bool)
		for i in s: #intersection
			tA = np.logical_and(maskM[i,:], tA)
		for i in list(setSigI-setS): #remove differences
			tA[maskM[i,:]] = 0
		rBounds = calcRegionBounds(tA)
		name = 'S'+'S'.join(map(str,sorted(np.array(s)+1)))
		for b,e in rBounds:
			OS.write("%s\t.\tgene\t%i\t%s\t.\t.\t.\tID=gene%04d;Name=%s;color=%s;\n" % (locations[i][b][0], int(locations[i][b][1])+1, locations[i][e][2], count, name, colorDict[frozenset(s)]))
			count += 1
	OS.close()
			
def powerSet(L):
	'''
	Removes the empty list from powerset

	>>> powerSet([1,2])
	[[1], [0], [1, 0]]
	'''
	def powerHelp(A):
		'''
		Builds powerset recursively.

		>>> powerHelp([1,2])
		[[], [1], [0], [1, 0]]
		'''
		if not A:
			return [[]]
		ret = powerHelp(A[1:])
		return ret+[i+[A[0]] for i in ret]
	return powerHelp(L)[1:]

def updateCount(countM, signals, sigI, sigV, sBins, pThresh):
	#countM[sample, index]
	#signals[index, sample]
	for locI, mixI in zip(sigI,sigV):
		mult = 1.0
		if mixI < 0:
			mult = -1.0
		outSignal = haarVec(hannVec(mult*signals[:,np.abs(mixI)], windowSize=sBins), p=pThresh)
		#outSignal = hannVec(haarVec(mult*signals[:,np.abs(mixI)], p=pThresh), windowSize=sBins)
		countM[locI,outSignal>zero] += 1

def writeSignals(signal,locations,name,index,sBins,pThresh,merge):
	outFile = "separated_signal_%i.bedgraph"%(index+1)
	outSignal = hannVec(haarVec(signal, p=pThresh), windowSize=sBins)
	print '\t'+outFile
	OF = open(outFile,'w')
	N = len(outSignal)
	for i in xrange(N):
		outStr = '%s\t%s\t%s\t%.6f\n' % tuple(locations[i]+[outSignal[i]])
		OF.write(outStr)
	OF.close()
	#write gff
	bMask = np.zeros(len(outSignal), dtype=np.bool)
	bMask[outSignal > zero] = 1
	mergeRegions(bMask, distThresh=merge)
	outGFF = 'separated_signal_%i.gff3'%(index+1)
	print '\t'+outGFF
	rBounds = calcRegionBounds(bMask)
	OF = open(outGFF,'w')
	OF.write('##gff-version 3\n#track name="S%i" gffTags=on\n' % (index+1))
	count = 1
	for s,e in rBounds:
		OF.write("%s\t.\tgene\t%i\t%s\t.\t.\t.\tID=gene%04d;color=%s;\n" % (locations[s][0], int(locations[s][1])+1, locations[e][2], count, myColors[index]))
		count += 1
	OF.close()

def pullVal(absMixing, mixing, index, sigV):
#def pullVal(mixing, index, pair):
#	mixVal = mixing[index,pair[index,1]]
#	if mixVal < 0:
#		return -1*pair[index,1]
#	return pair[index,1]
	absMax = np.nanargmax(absMixing[index])
	print "Index %i got compontent %i" % (index, absMax)
	if mixing[index,absMax] < 0:
		sigV.append(-absMax)
	else:
		sigV.append(absMax)
	absMixing[:,absMax] = np.nan

def processFiles(locations, vals, files):
	for f in files:
		l,v = parseBG(f)
		locations.append(l)
		vals.append(v)

def haarVec(vals, p=80):
	#print "Calculating Haar with p=%.2f"%(p)
	hC = pywt.wavedec(vals,'haar')
	cutVal = np.percentile(np.abs(np.concatenate(hC)), p)
	for A in hC:
		A[np.abs(A) < cutVal] = 0
	tVals = pywt.waverec(hC,'haar')
	return tVals[:len(vals)]

def hannVec(vals, windowSize=20):
	#print "Hann smoothing for %i bins" % (windowSize)
	# Smooths using a hanning window
	w = np.hanning(windowSize)
	return np.convolve(w/w.sum(), vals, mode='same')

def smoothData(vals,windowSize=20, p=80):
	# Smooths using a hanning window
	for i in xrange(vals.shape[0]):
		vals[i,:] = haarVec(vals[i,:], p=p)
		vals[i,:] = hannVec(vals[i,:], windowSize=windowSize)
	return vals

def normalize(vals):
	'''
	Normalize the counts using DESeq's method

	>>> np.round(normalize(np.array([[1,2,3],[4,5,6]])),2)
	array([[ 1.58,  3.16,  4.74],
	       [ 2.53,  3.16,  3.79]])
	'''
	sf = sizeFactors(vals)
	return np.array(vals/np.matrix(sf).T)
	
def parseBG(inFile):
	lines = open(inFile,'r').readlines()
	tmp = map(lambda y: y.rstrip('\n').split('\t'), lines)
	locs = map(lambda y: y[:3], tmp)
	vals = np.array(map(lambda y: np.float(y[3]), tmp))
	return locs, vals

def geometricMean(M):
	'''
	Returns the geometric mean of a numpy array.

	Parameters
	============================================
	M	numpy array (matrix)
	
	>>> geometricMean(np.array([[1,2,3],[4,5,6]]))
	array([ 2.        ,  3.16227766,  4.24264069])
	>>> geometricMean(np.array([[0.1, 1, 2],[10,1,0.2]]))
	array([ 1.        ,  1.        ,  0.63245553])
	'''
	return np.prod(M,axis=0)**(1.0/M.shape[0])

def sizeFactors(npVals):
	'''
	Calculates size factors like DESeq. http://genomebiology.com/2010/11/10/R106
	- equation 5

	Parameters
	============================================
	M	numpy array (matrix)

	>>> np.round(sizeFactors(np.array([[0.1,1,2],[10,1,0.2]])),2)
	array([ 1.,  1.])
	>>> sizeFactors(np.array([[-1,1],[0,3]]))
	Traceback (most recent call last):
	 ...
	ArithmeticError: Negative values in matrix
	>>> np.round(sizeFactors(np.array([[1,2,3],[4,5,6]])),2)
	array([ 0.63,  1.58])
	'''
	if np.any(npVals < 0):
		raise ArithmeticError("Negative values in matrix")
	piArray = geometricMean(npVals)
	gZero = piArray > 0
	return np.median(npVals[:,gZero]/piArray[gZero], axis=1)

def mergeRegions(counter, distThresh=0):
	'''
	Merges regions that are closer than (<) the distance threshold.

	Parameters
	=============================
	counter		Binary counter array
	distThresh	Max distance threshold

	>>> A=np.array([1,1,0,0,0,1,1])
	>>> mergeRegions(A)
	>>> A
	array([1, 1, 0, 0, 0, 1, 1])
	>>> mergeRegions(A, distThresh=3)
	>>> A
	array([1, 1, 0, 0, 0, 1, 1])
	>>> mergeRegions(A, distThresh=4)
	>>> A
	array([1, 1, 1, 1, 1, 1, 1])
	'''
	bounds = calcRegionBounds(counter)
	for i in xrange(len(bounds)-1):
		start0, end0 = bounds[i]
		start1, end1 = bounds[i+1]
		if start1-end0-1 < distThresh:
			counter[start0:end1] = 1

def calcRegionBounds(counter):
	'''
	Returns the new lower and upper bounds over overlapped regions.

	Parameters
	=============================
	counter		Binary counter array

	>>> calcRegionBounds(np.array([1,1,0,0,1,1,1,0,0,1,1]))
	array([[ 0,  1],
	       [ 4,  6],
	       [ 9, 10]])
	'''
	d = np.diff(counter)
	idx, = d.nonzero()
	if counter[0]:
		idx = np.r_[-1, idx]
	if counter[-1]:
		idx = np.r_[idx, counter.size-1]
	idx.shape = (-1,2)
	idx[:,0] += 1
	return idx

if __name__ == "__main__":
	main()

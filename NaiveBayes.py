import numpy as np
import csv
import math

def loadFile(fileName):
	lines = csv.reader(open(fileName,"rb"))
	data = list(lines)
	for i in range(len(data)):
		data[i] = [float(v) for v in data[i]]  
	return data
def splitData(dataset):
	split = 0.8 
	np.random.shuffle(dataset)
	trainData = dataset[:int(len(dataset)*split)]
	testData = dataset[int(len(dataset)*split):]
	return [trainData,testData]

def separateData(data):
	''' This separation only applies to the pima indians diabetes dataset'''
	'''
	sepData = {}
	sepData = [map(float,item[:]) for item in data if item[8] == 1]
	
	vector = {}
	vector = [map(float,item[:]) for item in data if item[8] == 0] 
	'''
	''' vector = {} 
	for i in range(len(data)):
		if(data[i] not in sepData):
			sepData.append(data[i]) 
	'''
	'''
	sdata = {}
	sdata.append(sepData)
	'''
	'''for i in range(len(vector)):'''
	'''
	sdata.append(vector) 
	return sdata
	'''
	'''return [sepData,vector] ''' 
	sepData = {}
	for i in range(len(data)):
		vector = dataset[i]
		if(vector[-1] not in sepData):
			sepData[vector[-1]] = []
		sepData[vector[-1]].append(vector)
	return sepData


def mean(x):
	return sum(x)/float(len(x))

def standardDeviation(x):
	avg = mean(x)
	variance = sum([pow(v-m,2) for v in x])/float(len(x)-1)
	return math.sqrt(variance)

def summary(dataset):
	summaries = [(mean(v),standardDeviation(v)) for v in zip(*dataset)]
	del summaries[-1]
	return summaries

def summaryData(data):
	separated = separateData(data)
	su = {}
	for cv, instances in separated.items():
		su[cv] = summary(instances)
	return su

def getProb(v,mean,stdev):
	ex = math.exp(-(math.pow(v-mean,2)/(2*math.pow(stdev,2))))
	return (1/(math.sqrt(2*math.pi)*stdev)) * ex

def getClassProb(summaries, vec):
	prob = {}
	for cv, classSummaries in summaries.items():
		prob[cv] =  1
		for i in range(len(classSummaries)):
			mean, stdev = classSummaries[i]
			x = vec[i]
			prob[cv] = prob[cv]*getProb(x,mena,stdev)
	return prob

def predict(summaries, vec):
	prob = getClassProb(summaries, vec)
	label, bestprob = None, -1
	for cv,p in prob.items():
		if label is None or p > bestprob:
			bestprob = p
			label = cv
	return label

def getPredictions(summaries, test):
	predictions = []
	for i in range(len(test)):
		res = predict(summaries,test[i])
		predictions.append(res)
	return predictions

def accuracy(test, predictions):
	acc = 0.0
	count = 0
	for v in range(len(test)):
		if(test[v][-1] == predictions[v]):
			count = count+1
	acc = (count/float(len(test))) *100.0
	return acc

def main():
	fileName = "pima-indians-diabetes.data.csv"
	data = loadFile(fileName)
	trainSet, testSet = splitData(data)
	summaries = summaryData(trainSet)
	predictions = getPredictions(summaries, testSet)
	acc = accuracy(testSet,predictions)
	print("Accuracy is: {0}%").format(acc)

main()

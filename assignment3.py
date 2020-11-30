#Xavier Vargas
#Machine Learning Assignment 3
import sys
import random

def distanceSquared(x,y):
    result=0
    for i in range(len(x)):
        num = int(y[i])-int(x[i])
        num *= num
        result+=num
    return result

def classify(exemplars,instance):#returns v, the exemplar closest to the instance
    distance = float('inf')
    for exemplar in exemplars:
        dist = distanceSquared(exemplars[exemplar],instance)
        if dist < distance:
            distance = dist
            v = exemplar
    return v

def gradient(c): #c is a variable length string
    retArray =[]
    pCount =0
    for k in range(len(c)):
        pCount+=1
        pLen = len(c[k])
        for i in range(pLen):
            if pCount == 1:
                retArray.append(int(c[k][i]))
            else:
                retArray[i] += int(c[k][i])
    for j in range(len(retArray)):
        sum = retArray[j]
        fill = sum/pCount
        retArray[j] = fill
    return retArray
def computAccuracy(exemplars,domC):#classify the Table given the set of exemplars, then test how many we got right.
    classifications =[]
    correct =0
    wrong =0
    for c in domC:
        for point in domC[c]:
            classification = classify(exemplars,point)
            if classification != c:
                wrong +=1
            else:
                correct +=1
    return correct/(correct+wrong)
def average(array):
    sum = 0
    for elem in array:
        sum+=int(elem)
    return sum/len(array)
def initTrainingSet(fileName):
    f = open(fileName,"r")
    domC = dict() # keys are classifiers, value is set of points
    fullTable = []
    for line in f:
        point = line.split(",")
        value =[]
        for elem in point:
            if elem.strip(" \n").isalpha():
                key = elem.strip("\n")
            else:
                value.append(float(elem))
        if key in domC:
            domC[key].append(value)
        else:
            domC[key] = [value]#might have to change this to -> domC[point[3]] = domC[point[3]].add(point[:3])
        fullTable.append(point)
    f.close()
    return fullTable, domC
def initGradientVectors(domC):
    gradients = dict()
    for c in domC:
        gradients[c] = gradient(domC[c])
    return gradients
def randomGradientVectors(Table,domC):
    hardmax = float('inf')
    # attributeMin =[hardmax,hardmax,hardmax]
    # attributeMax =[0,0,0]
    attributeMin =[]
    attributeMax =[]
    pCount =0
    for point in Table:
        pCount+=1
        for number in range(len(point)-1):
            if pCount == 1:
                attributeMin.append(hardmax)
                attributeMax.append(0)
            if float(point[number]) < attributeMin[number]:
                attributeMin[number] = float(point[number])
            elif float(point[number]) > attributeMax[number]:
                attributeMax[number] = float(point[number])
    newVectors =dict()
    for c in domC:
        newVector = []
        for i in range(len(attributeMax)):
            newVector.append(random.uniform(attributeMin[i],attributeMax[i]))
        newVectors[c] = newVector
    return newVectors
def gradDescent(domC,vectors,stepSize,epsilon,M):
    PrevCost = float('inf')
    PrevAccuracy = computAccuracy(vectors,domC)
    while (True):
        totalCost = 0.0
        bigN = dict()
        for v in domC:
            bigN[v] = []
            for i in range(len(vectors[v])):
                bigN[v].append(0)
        #now go through each instance in T, which we can access equally by going through each set in domC domC = dict of key/value being classification/ set of points
        for v in domC:
            for dataPoint in domC[v]:
                classification = classify(vectors,dataPoint)
                gsubW = vectors[classification]
                if classification != v:
                    cost = distanceSquared(gradient(domC[v]), dataPoint) - distanceSquared(dataPoint, gsubW)
                    if cost < M:
                        for i in range(len(bigN[v])):
                            bigN[v][i] = bigN[v][i] + (int(dataPoint[i]) - int(vectors[v][i]))
                        for i in range(len(gsubW)):
                            bigN[classification][i] = bigN[classification][i]+(int(gsubW[i])-int(dataPoint[i]))
                    else:
                        cost += M
        if totalCost < epsilon:
            return vectors
        if totalCost > (1- epsilon)*PrevCost:
            return vectors
        bigH = dict()
        for exemplar in vectors:
            for i in range(len(vectors[exemplar])):
                bigH[exemplar] = vectors[exemplar][i] + stepSize * bigN[exemplar][i]
        newAccuracy = computAccuracy(bigH,domC)
        if newAccuracy < PrevAccuracy:
            return vectors
        for v in domC:
            vectors[v] = bigH[v]
        PrevCost = totalCost
        PrevAccuracy = newAccuracy

argc = len(sys.argv)

if argc == 7:
    verboseOutput = True
    trainingFile = sys.argv[2]
    stepSize = float(sys.argv[3])
    epsilon = float(sys.argv[4])
    M = float(sys.argv[5])
    randomRestarts = sys.argv[6]
else:
    verboseOutput = False
    trainingFile = sys.argv[1]
    stepSize = float(sys.argv[2])
    epsilon = float(sys.argv[3])
    M = float(sys.argv[4])
    randomRestarts = sys.argv[5]
Table, domC = initTrainingSet(trainingFile)
bestAccuracy = 0.0
bestExemplars = None
for i in range(int(randomRestarts)):#replace 2 with number of random restarts
    if i == 0:
        vectors = initGradientVectors(domC)
        ff = open("VX_Output.txt","w")
    else:
        vectors = randomGradientVectors(Table,domC)
    ff.write("Iteration:"+str(i)+"\n")
    testExemplars = gradDescent(domC,vectors,stepSize,epsilon,M)
    if verboseOutput:
        for exemplar in testExemplars:
            for i in range(len(testExemplars[exemplar])):
                if i == len(testExemplars[exemplar])-1:
                    var = testExemplars[exemplar][i]
                    ff.write(str(round(var,3))+"\n")
                else:
                    var = testExemplars[exemplar][i]
                    ff.write(str(round(var, 3)) + ", ")
            # print(testExemplars[exemplar])
    accurateMeasure = computAccuracy(testExemplars,domC)
    ff.write("Accuracy "+str(round(accurateMeasure,4))+"\n")
    ff.write("\n")
    if accurateMeasure > bestAccuracy:
        bestAccuracy = accurateMeasure
        bestExemplars = testExemplars
ff.write("Best accuracy: "+str(round(bestAccuracy,2))+"\n")
# ff.write("Exemplars used:"+str(bestExemplars))
ff.close()

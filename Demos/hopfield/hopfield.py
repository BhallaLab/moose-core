#PARTIAL DEMO
# demo of a simple hopfield network using leaky integrate and fire neurons
# memory.csv has the memory being saved, the synaptic weights are set at start according to this memory
# memory must be a square matrix of 0's and 1's only
# input.csv has partial input given as input to the cells.
# - By C.H.Chaitanya
# This code is available under GPL3 or greater GNU license 

import moose
import csv
import numpy as np

memoryFile = "memory.csv"
inputFile = "input.csv"

def readMemory(memoryFile):
    f = open(memoryFile,'r')
    testLine = f.readline()
    dialect = csv.Sniffer().sniff(testLine) #to get the format of the csv
    f.close()

    f = open(memoryFile, 'r')
    reader = csv.reader(f,dialect)
    memory = []
    for row in reader:
        for i in row[0:]:
            memory.append(int(i))
    f.close()
    return memory

def updateWeights(memory,weightMatrix):
    for i in range(len(memory)):
        tempList = []
        for j in range(len(memory)):
            if i != j:
                tempList.append(memory[i]*memory[j])
            else:
                tempList.append(0)
        weightMatrix.append(tempList)
    return weightMatrix

def createNetwork(synWeights):
    numberOfCells = len(synWeights[0])
    cells = []
    for i in range(numberOfCells):
        cell = moose.IntFire('/cell_'+str(i+1)) #non programmer friendly numbering
        cell.setField('tau',10.0)
        cell.setField('Vm', -0.07)
        #cell.setField('refractoryPeriod', 0.1)
        #cell.setField('thresh', 0.0)

        inSyn = moose.element('/cell_'+str(i+1)+'/synapse')
        inSpkGen = moose.SpikeGen('/cell_'+str(i+1)+'/inSpkGen')
        moose.connect(inSpkGen, 'event', inSyn ,'addSpike') 

        cells.append(cell)

    for currentCell in range(numberOfCells): #currCell 
        for connectCell in range(numberOfCells): #connectToCell
            if currentCell != connectCell:
                #how to give synaptic input weight? synWeights[currentCell][connectCell]
                connSyn = moose.element('/cell_'+str(connectCell+1)+'/synapse')
                moose.connect(moose.element('/cell_'+str(currentCell+1)), 'spike', connSyn, 'addSpike')

    return cells

memory = readMemory(memoryFile)

synWeights = updateWeights(memory,[])

cells = createNetwork(synWeights)


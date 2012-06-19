from moose import *
import sys
import matplotlib.pyplot as plt

class plot:
    def __init__(self,searchpath):
        for x in wildcardFind(searchpath):
            plt.plot(moose.Table(x).vec)
        plt.show()

if __name__ == '__main__':
    modelPath = sys.argv[1]
    loadPath = modelPath[modelPath.rfind('/'):modelPath.rfind('.')]
    runtime = sys.argv[2] 
    loadModel(modelPath,loadPath)
    searchpath = loadPath+'/graphs/#/##[TYPE=Table],'+loadPath+'/moregraphs/#/##[TYPE=Table]'
    reinit()
    start (float(runtime))
    plot(searchpath)

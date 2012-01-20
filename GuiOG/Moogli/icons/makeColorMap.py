import numpy 
import pickle; import os; import sys
from matplotlib import pyplot

def makeColorMapPNG(fileName):
    f = open(fileName,'r')
    l = pickle.load(f)
    width = len(l)
    height = 20
    a = numpy.zeros([height,width,3])

    for i in range(height):
        for j in range(width):
            for k in range(3):
                a[i][j][k] = l[j][k]

    fig=pyplot.imshow(a)
    pyplot.xticks([])
    pyplot.yticks([])
    pyplot.savefig(os.path.split(fileName)[1]+'.png',transparent='True',bbox_inches='tight',dpi=50)

makeColorMapPNG(sys.argv[1])

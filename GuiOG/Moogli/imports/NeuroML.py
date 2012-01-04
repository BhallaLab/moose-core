from xml.etree import ElementTree as ET
#from ChannelML import *
from MorphML import *
#from NetworkML import *
import string
import sys

class NeuroML():

    def __init__(self):
        self.neuroml_ns='http://morphml.org/neuroml/schema'

    def readNeuroMLFromFile(self,filename,params={}):
        """
        For the format of params required to tweak what cells are loaded,
        refer to the doc string of NetworkML.readNetworkMLFromFile().
        """
 
        tree = ET.parse(filename)
        root_element = tree.getroot()
        self.length_units = root_element.attrib['length_units']
        print filename
        mmlR = MorphML()
        self.cellsList = []
        for cells in root_element.findall('.//{'+self.neuroml_ns+'}cells'):
            print 'found a cell'
            for cell in cells.findall('.//{'+self.neuroml_ns+'}cell'):
                cellList = mmlR.readMorphML(cell,params={},length_units=self.length_units)
                self.cellsList.extend(cellList)
        return self.cellsList

def loadNeuroML_L123(filename):
    neuromlR = NeuroML()
    neuromlR.readNeuroMLFromFile(filename)    

if __name__ == "__main__":
    if len(sys.argv)<2:
        print "You need to specify the neuroml filename."
        sys.exit(1)
    loadNeuroML_L123(sys.argv[1])

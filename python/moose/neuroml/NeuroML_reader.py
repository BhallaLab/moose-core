from xml.etree import ElementTree as ET
from ChannelML_reader import *
from MorphML_reader import *
from NetworkML_reader import *
import string
import moose
import sys

class NeuroML():

    def __init__(self):
        self.context = moose.PyMooseBase.getContext()

    def readNeuroMLFromFile(self,filename):
        print "Loading neuroml file ... ", filename
        tree = ET.parse(filename)
        root_element = tree.getroot()
        self.lengthUnits = root_element.attrib['lengthUnits']

        #print "Loading channels and synapses into MOOSE /library ..."
        cmlR = ChannelML()
        for channels in root_element.findall('.//{'+neuroml_ns+'}channels'):
            self.channelUnits = channels.attrib['units']
            for channel in channels.findall('.//{'+cml_ns+'}channel_type'):
                ## ideally I should read in extra params
                ## from within the channel_type element and put those in also.
                ## Global params should override local ones.
                cmlR.readChannelML(channel,params={},units=self.channelUnits)
            for synapse in channels.findall('.//{'+cml_ns+'}synapse_type'):
                cmlR.readSynapseML(synapse,units=self.channelUnits)
            for ionConc in channels.findall('.//{'+cml_ns+'}ion_concentration'):
                cmlR.readIonConcML(ionConc,units=self.channelUnits)

        #print "Loading cell definitions into MOOSE /library ..."
        mmlR = MorphML()
        self.cellsDict = {}
        for cells in root_element.findall('.//{'+neuroml_ns+'}cells'):
            for cell in cells.findall('.//{'+neuroml_ns+'}cell'):
                cellDict = mmlR.readMorphML(cell,params={},lengthUnits=self.lengthUnits)
                self.cellsDict.update(cellDict)

        #print "Loading individual cells into MOOSE root ... "
        nmlR = NetworkML()
        self.populationDict, self.projectionDict = \
            nmlR.readNetworkML(root_element,self.cellsDict,params={},lengthUnits=self.lengthUnits)

def loadNeuroML_L123(filename):
    neuromlR = NeuroML()
    neuromlR.readNeuroMLFromFile(filename)    

if __name__ == "__main__":
    if len(sys.argv)<2:
        print "You need to specify the neuroml filename."
        sys.exit(1)
    loadNeuroML_L123(sys.argv[1])

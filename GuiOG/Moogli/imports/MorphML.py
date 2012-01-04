from xml.etree import ElementTree as ET
import string
import sys, math

#from ChannelML import *
#from moose.utils import *

class MorphML():

    def __init__(self):
        self.neuroml='http://morphml.org/neuroml/schema'
        self.bio='http://morphml.org/biophysics/schema'
        self.mml='http://morphml.org/morphml/schema'
        self.nml='http://morphml.org/networkml/schema'

    def readMorphMLFromFile(self,filename,params={}):
        tree = ET.parse(filename)
        neuroml_element = tree.getroot()

        if neuroml_element.tag.rsplit('}')[1] == 'neuroml':
            cellTag = self.neuroml
        else:
            cellTag = self.mml

        cellsList = [] #chk name space of root element = neuroml
        for cell in neuroml_element.findall('.//{'+cellTag+'}cell'):
            cellList = self.readMorphML(cell,params,neuroml_element.attrib['length_units'])
            cellsList.extend(cellList)
        return cellsList

    def readMorphMLFromString(self,filename,params={}):
        tree = ET.ElementTree(ET.fromstring(filename))
        neuroml_element = tree.getroot()

        if neuroml_element.tag.rsplit('}')[1] == 'neuroml':
            cellTag = self.neuroml
        else:
            cellTag = self.mml

        cellsList = [] #chk name space of root element = neuroml
        for cell in neuroml_element.findall('.//{'+cellTag+'}cell'):
            cellList = self.readMorphML(cell,params,neuroml_element.attrib['length_units'])
            cellsList.extend(cellList)
        return cellsList

    def readMorphML(self,cell,params={},length_units="micrometer"):
        if length_units in ['micrometer','micron']:
            self.length_factor = 1e-06
        else:
            self.length_factor = 1.0
        cellname = cell.attrib["name"]
        compartmentList = []
        proximalDict = {}
        #### load morphology and connections between compartments
        for segment in cell.findall(".//{"+self.mml+"}segment"):

            segmentname = segment.attrib['name']
            segmentid = segment.attrib['id']

            compartmentList.append([segmentname,cellname])
            if segment.attrib.has_key('parent'):
                parentid = segment.attrib['parent']
            proximal = segment.find('./{'+self.mml+'}proximal')
            if proximal == None:
                x0 = proximalDict[parentid][0]
                y0 = proximalDict[parentid][1]
                z0 = proximalDict[parentid][2]
            else:    
                x0 = float(proximal.attrib["x"])*self.length_factor
                y0 = float(proximal.attrib["y"])*self.length_factor
                z0 = float(proximal.attrib["z"])*self.length_factor

            distal = segment.find('./{'+self.mml+'}distal')
            x = float(distal.attrib["x"])*self.length_factor
            y = float(distal.attrib["y"])*self.length_factor
            z = float(distal.attrib["z"])*self.length_factor
            proximalDict[segmentid] = [x,y,z] #incase child element 
            diameter = float(distal.attrib["diameter"]) * self.length_factor

            currentIndex = len(compartmentList)-1
            compartmentList[currentIndex].append([x0,y0,z0,x,y,z,diameter]) 

        return compartmentList

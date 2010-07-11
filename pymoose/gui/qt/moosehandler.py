# moosehandler.py --- 
# 
# Filename: moosehandler.py
# Description: 
# Author: subhasis ray
# Maintainer: 
# Created: Thu Jan 28 15:08:29 2010 (+0530)
# Version: 
# Last-Updated: Sun Jul 11 13:31:56 2010 (+0530)
#           By: Subhasis Ray
#     Update #: 698
# URL: 
# Keywords: 
# Compatibility: 
# 
# 

# Commentary: 
# 
# 
# 
# 

# Change log:
# 
# 
# 
# 
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation; either version 3, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program; see the file COPYING.  If not, write to
# the Free Software Foundation, Inc., 51 Franklin Street, Fifth
# Floor, Boston, MA 02110-1301, USA.
# 
# 

# Code:

import os
import sys
import random
from collections import defaultdict
import re
import xml.sax as sax
import xml.sax.handler as saxhandler
import xml.sax.xmlreader as saxreader
import xml.sax.saxutils as saxutils

from PyQt4 import QtCore
import moose
import config
from glclient import GLClient


class MooseXMLHandler(saxhandler.ContentHandler):
    def __init__(self):
        saxhandler.ContentHandler.__init__(self)
        self.model_type = None

    def startElement(self, name, attrs):
        """Signal the start of an element.

        This method looks for neuroml/sbml tags to recognize the type of the model.
        """
        if name == 'sbml':
            self.model_type = MooseHandler.type_sbml
        elif name == 'neuroml':
            self.model_type = MooseHandler.type_neuroml
        else:
            pass

class MooseHandler(QtCore.QObject):
    """Access to MOOSE functionalities"""
    # A list keys for known filetypes Note that type_genesis includes
    # kkit (both have same extension and we separate them only after
    # looking for 'include kkit' statement inside the file. Similarly,
    # both type_neuroml and type_sbml are of type_xml. We recognise
    # the exact type only after looking inside the file.
    type_genesis = 'GENESIS'
    type_kkit = 'KKIT'
    type_xml = 'XML'
    type_neuroml = 'NEUROML'
    type_sbml = 'SBML'
    type_python = 'PYTHON'
    # Map between file extension and known broad filetypes.
    fileExtensionMap = {
        'Genesis Script(*.g)': type_genesis,
        'neuroML/SBML(*.xml *.bz2 *.zip *.gz)': type_xml,
        'Python script(*.py)': type_python
        }
    DEFAULT_SIMDT = 2.5e-5
    DEFAULT_PLOTDT = 1e-4
    DEFAULT_GLDT = 5e-4
    DEFAULT_RUNTIME = 1e-1
    DEFAULT_PLOTUPDATE_DT = 1e-1
    simdt = DEFAULT_SIMDT
    plotdt = DEFAULT_PLOTDT
    gldt = DEFAULT_GLDT
    runtime = DEFAULT_RUNTIME
    plotupdate_dt = DEFAULT_PLOTUPDATE_DT
    def __init__(self):
        QtCore.QObject.__init__(self)
	self._context = moose.PyMooseBase.getContext()
	self._lib = moose.Neutral('/library')
	self._proto = moose.Neutral('/proto')
	self._data = moose.Neutral('/data')
        self._gl = moose.Neutral('/gl')
        self._current_element = moose.Neutral('/')
        self._xmlreader = sax.make_parser()
        self._saxhandler = MooseXMLHandler()
        self._xmlreader.setContentHandler(self._saxhandler)
        self.fieldTableMap = {}
        self._tableIndex = 0
        self._tableSuffix = random.randint(1, 999)
        self._connSrcObj = None
        self._connDestObj = None
        self._connSrcMsg = None
        self._connDestMsg = None
        # The follwoing maps for managing 3D visualization objects
        self._portClientMap = {}
        self._portPathMap = {}
        self._pathPortMap = defaultdict(set)
        self._portServerMap = {}

    def runGenesisCommand(self, cmd):
	"""Runs a GENESIS command and returns the output string"""
	self._context.runG(cmd)
        return 'In current PyMOOSE implementation running a GENESIS command does not return anything.'

    def loadModel(self, filename, filetype):
        """Load a model from file."""
        directory = os.path.dirname(filename)
        os.chdir(directory)
        filename = os.path.basename(filename) # ideally this should not be required - but neuroML reader has a bug and gets a segmentation fault when given abosolute path.
        moose.Property.addSimPath(directory)
        if filetype == MooseHandler.type_genesis:
            return self.loadGenesisModel(filename)
        elif filetype == MooseHandler.type_xml:
            return self.loadXMLModel(filename)
        elif filetype == MooseHandler.type_python:
            sys.path.append(directory)
            return self.loadPythonScript(filename)
        


    def loadGenesisModel(self, filename):
        """Load a model specified in a GENESIS Script.

        If the file is a kinetikit model (the criterion is 'include
        kkit' statement somewhere near the beginning of the file, it
        returns MooseHandler.type_kkit.

        Returns MooseHandler.type_genesis otherwise.
        """
        filetype = MooseHandler.type_genesis
        kkit_pattern = 'include *kkit'
        in_comment = False
        with open(filename, 'r') as infile:
            sentence = ''
            in_sentence = False
            for line in infile:
                line = line.strip()
                if line.find('//') == 0: # skip c++ style comments
                    continue
                comment_start = line.find('/*')
                if comment_start >= 0:
                    in_comment = True
                sentence = line[:comment_start] 
                while in_comment and line:
                    comment_end = line.find('*/')
                    if comment_end >= 0:
                        in_comment = False
                        sentence = sentence + line[comment_end+2:] # add the rest of the line to sentence
                    line = infile.readline()
                    line = line.strip()
                if in_sentence:
                    sentence = sentence + line.strip('\\')
                else:
                    sentence = ''
                if line and line.endswith('\\'):
                    in_sentence = True                    
                if re.search(kkit_pattern, sentence):
                    filetype = MooseHandler.type_kkit
                    break
        self._context.loadG(filename)
        return filetype
        
    def loadXMLModel(self, filename):
        """Load a model in some XML format. 

        Looks inside the XML to figure out if this is a neuroML or an
        SBML file and calls the corresponding loader functions.

        Currently only SBML and neuroML are support. In future 9ml
        support will be provided as the specification becomes stable.

        """
        with open(filename, 'r') as xmlfile:
            for line in xmlfile:
                self._xmlreader.feed(line)
                if self._saxhandler.model_type is not None:
                    break
        ret = self._saxhandler.model_type
        self._saxhandler.model_type = None
        self._xmlreader.reset()
        if ret == MooseHandler.type_neuroml:
            command = 'readNeuroML "%s" %s' % (filename, self._current_element.path)
            self._context.readNeuroML(filename, self._current_element.path)
        elif ret == MooseHandler.type_sbml:
            self._context.readSBML(filename, self._current_element.path)
        
        return ret    

    def loadPythonScript(self, filename):
        """Evaluate a python script."""
        extension_start = filename.rfind('.py')
        script = filename[:extension_start]
        exec 'import %s' % (script)

    def doReset(self, simdt, plotdt, gldt, plotupdate_dt):
        """Reset moose.

        simdt -- dt for simulation (step size for numerical
        methods. Clock tick 0, 1 and 2 will have this dt.

        plotdt -- time interval for recording data.

        gldt -- time interval for OpenGL display.

        We put all the table objects under /data on clock 3, all the
        GLcell and GLview objects under /gl on clock 4.
        
        """
        self._context.setClock(0, simdt)
        self._context.setClock(1, simdt)
        self._context.setClock(2, simdt)
        self._context.setClock(3, plotdt)
        self._context.setClock(4, gldt)
        self._context.useClock(3, self._data.path + '/##[TYPE=Table]')
        self._context.useClock(4, self._gl.path + '/##[TYPE=GLcell]')
        self._context.useClock(4, self._gl.path + '/##[TYPE=GLview]')
        MooseHandler.simdt = simdt
        MooseHandler.plotdt = plotdt
        MooseHandler.gldt = gldt
        MooseHandler.plotupdate_dt = plotupdate_dt
        self._context.reset()

    def addFieldTable(self, full_field_path):
        """
        adds a field to the list of fields to be plotted.

        full_field_path -- complete path to the field.
        """
        try:
            table = self.fieldTableMap[full_field_path]
        except KeyError:
            fstart = full_field_path.rfind('/')
            fieldName = full_field_path[fstart+1:]
            objPath =  full_field_path[:fstart]
            table = moose.Table('%s_%d_%d' % (fieldName, self._tableSuffix, self._tableIndex), self._data)
            self.fieldTableMap[full_field_path] = table
            table.stepMode = 3
            target = moose.Neutral(objPath)
            connected = table.connect('inputRequest', target, fieldName)
            config.LOGGER.info('Connected %s to %s/%s' % (table.path, target.path, fieldName))
            self._tableIndex += 1
        return table

    def doRun(self, time):
        """Just runs the simulation. 

        If time is float, it is absolute time in seconds.
        If an integer, it is the number of time steps.

        """
        MooseHandler.runtime = time      
        next_stop = MooseHandler.plotupdate_dt
        while next_stop <= MooseHandler.runtime:
            self._context.step(MooseHandler.plotupdate_dt)
            next_stop = next_stop + MooseHandler.plotupdate_dt
            self.emit(QtCore.SIGNAL('updatePlots(float)'), self._context.getCurrentTime())
        time_left = MooseHandler.runtime + MooseHandler.plotupdate_dt - next_stop 
        if MooseHandler.runtime < MooseHandler.plotupdate_dt:
            time_left = MooseHandler.runtime
        self._context.step(time_left)
        self.emit(QtCore.SIGNAL('updatePlots(float)'), self._context.getCurrentTime())
        
    def doConnect(self):
        ret = False
        if self._connSrcObj and self._connDestObj and self._connSrcMsg and self._connDestMsg:
            ret = self._connSrcObj.connect(self._connSrcMsg, self._connDestObj, self._connDestMsg)
            print 'Connected %s/%s to %s/%s: ' % (self._connSrcObj.path, self._connSrcMsg, self._connDestObj.path, self._connDestMsg), ret
            self._connSrcObj = None
            self._connDestObj = None
            self._connSrcMsg = None
            self._connDestMsg = None

    def setConnSrc(self, fieldPath):
        pos = fieldPath.rfind('/')
        moosePath = fieldPath[:pos]
        field = fieldPath[pos+1:]
        self._connSrcObj = moose.Neutral(moosePath)
        self._connSrcMsg = field

    def setConnDest(self, fieldPath):
        pos = fieldPath.rfind('/')
        moosePath = fieldPath[:pos]
        field = fieldPath[pos+1:]
        self._connDestObj = moose.Neutral(moosePath)
        self._connDestMsg = field

    def getSrcFields(self, mooseObj):
        srcFields = self._context.getFieldList(mooseObj.id, moose.FTYPE_SOURCE)
        sharedFields = self._context.getFieldList(mooseObj.id, moose.FTYPE_SHARED)
        ret = []
        for field in srcFields:
            ret.append(field)
        for field in sharedFields:
            ret.append(field)
        return ret

    def getDestFields(self, mooseObj):
        destFields = self._context.getFieldList(mooseObj.id, moose.FTYPE_DEST)
        sharedFields = self._context.getFieldList(mooseObj.id, moose.FTYPE_SHARED)
        ret = []
        for field in destFields:
            ret.append(field)
        for field in sharedFields:
            ret.append(field)
        return ret

    def makeGLCell(self, mooseObjPath, port, field=None, threshold=None, lowValue=None, highValue=None, vscale=None, bgColor=None, sync=None):
        """Make a GLcell instance.
        
        mooseObjPath -- path of the moose object to be monitored
        
        port -- string representation of the port number for the client.
        
        field -- name of the field to be observed. Vm by default.

        threshold -- the % change in the field value that will be
        taken up for visualization. 1% by default.

        highValue -- value represented by the last line of the
        colourmap file. Any value of the field above this will be
        represented by the colour corresponding to this value.

        lowValue -- value represented by the first line of the
        colourmap file. Any value of the field below this will be
        represented by the colour corresponding to this value.

        vscale -- Scaling of thickness for visualization of very thin
        compartments. 

        bgColor -- background colour of the visualization window.

        sync -- Run simulation in sync with the visualization. If on,
        it may slowdown the simulation.
        
        """
        glCellPath = mooseObjPath.replace('/', '_')  + str(random.randint(0,999))
        glCell = moose.GLcell(glCellPath, self._gl)
        glCell.useClock(4)
        glCell.vizpath = mooseObjPath
        glCell.port = port
        self._portPathMap[port] = mooseObjPath
        self._pathPortMap[mooseObjPath].add(port)        
        self._portServerMap[port] = glCell

        if field is not None:
            glCell.attribute = field
        if threshold is not None:
            glCell.threhold = threshold
        if highValue is not None:
            glCell.highvalue = highValue
        if lowValue is not None:
            glCell.lowvalue = lowValue
        if vscale is not None:
            glCell.vscale = vscale
        if bgColor is not None:
            glCell.bgcolor = bgColor
        if sync is not None:
            glCell.sync = sync
        print 'Created GLCiew for object', mooseObjPath, ' on port', port

        
    def makeGLView(self, mooseObjPath, port, fieldList, minValueList, maxValueList, colorFieldIndex, morphFieldIndex=None, grid=None, bgColor=None, sync=None):
        """
        Make a GLview object to visualize some field of a bunch of
        moose elements.
        
        mooseObjPath -- GENESIS-style path for elements to be
        observed.

        port -- port to use for communicating with the client.

        fieldList -- list of fields to be observed.

        minValueList -- minimum value for fields in fieldList. 

        maxValueList -- maximum value for fields in fieldList.
        
        colorFieldIndex -- index of the field to be represented by the
        colour of the 3-D shapes in visualization.

        morphFieldIndex -- index of the field to be represented by the
        size of the 3D shape.
        
        grid -- whether to put the 3D shapes in a grid or to use the
        x, y, z coordinates in the objects for positioning them in
        space.

        bgColor -- background colour of visualization window.

        sync -- synchronize simulation with visualization.

        """
        glViewPath = mooseObjPath.replace('/', '_') + str(random.randint(0,999))
        glView = moose.GLview(glViewPath, self._gl)
        glView.useClock(4)
        self._portPathMap[port] = mooseObjPath
        self._pathPortMap[mooseObjPath].add(port)        
        self._portServerMap[port] = glView
        glView.vizpath = mooseObjPath
        glView.clientPort = port
        if len(fieldList) > 5:
            fieldList = fieldList[:5]

        for ii in range(len(fieldList)):
            visField = 'value%dField' % (ii+1)
            setattr(glView, visField, fieldList[ii])
            try:
                setattr(glView, 'value%dmin' % (ii+1), minValueList[ii])
                setattr(glView, 'value%dmax' % (ii+1), maxValueList[ii])
            except IndexError:
                break
        glView.color_val = int(colorFieldIndex)
        if morphFieldIndex is not None:
            glView.morph_val = int(morphFieldIndex)
        if grid and (grid != 'off'):
            glView.grid = 'on'

        if bgColor is not None:
            glView.bgcolor = bgColor

        if sync and (sync != 'off'):
            glView.sync = 'on'

        print 'Created GLView', glView.path, 'for object', mooseObjPath, ' on port', port


    def startGLClient(self, executable, port, mode, colormap):
        """Start the glclient subprocess. 

        executable -- path to the client program.

        port -- network port used to communicate with the server.

        mode -- The kind of moose GL-element to interact with (glcell or
        glview)
        
        colormap -- path to colormap file for 3D rendering.
        
        """
        client = GLClient(executable, port, mode, colormap)
        self._portClientMap[port] = client
        print 'Created GLclient on port', port
        return client
        

    def stopGLClientOnPort(self, port):
        """Stop the glclient process listening to the specified
        port."""
        try:
            client = self._portClientMap.pop(port)
            client.stop()
        except KeyError:
            config.LOGGER.error('%s: port not used by any client' % (port))

    def stopGLClientsOnObject(self, mooseObject):
        """Stop the glclient processes listening to glcell/glview
        objects on the specified moose object."""
        path = mooseObject.path
        try:
            portSet = self._pathPortMap.pop(path)
            for port in portSet:
                self.stopGLClientOnPort(port)
        except KeyError:
            config.LOGGER.error('%s: no 3D visualization clients for this object.' % (path))

    def stopGL(self):
        """Make the dt of the clock on GLview and GLcell objects very
        large. Kill all the GLClient processes"""
        self._context.setClock(4, 1e10)
        for port, client in self._portClientMap.items():
            client.stop()
        
    
            
        

    
    
# 
# moosehandler.py ends here

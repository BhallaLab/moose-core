# moosehandler.py --- 
# 
# Filename: moosehandler.py
# Description: 
# Author: subhasis ray
# Maintainer: 
# Created: Thu Jan 28 15:08:29 2010 (+0530)
# Version: 
# Last-Updated: Fri Feb 25 11:18:27 2011 (+0530)
#           By: Subhasis Ray
#     Update #: 868
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

from __future__ import with_statement

import os
import sys
import random
from collections import defaultdict
import re
import xml.sax as sax
import xml.sax.handler as saxhandler
import xml.sax.xmlreader as saxreader
import xml.sax.saxutils as saxutils

from moose.neuroml.NeuroML import NeuroML
from PyQt4 import QtCore
import moose
import config


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
        elif name == 'neuroml' or name == 'networkml':
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
    type_all = 'allType'
    # Map between file extension and known broad filetypes.
    fileExtensionMap = {
        'All Supported(*.g *.xml *.bz2 *.zip *.gz *.py)': type_all,
        'Genesis Script(*.g)': type_genesis,
        'neuroML/SBML(*.xml *.bz2 *.zip *.gz)': type_xml,
        'Python script(*.py)': type_python
        }
    DEFAULT_SIMDT = 2e-5
    DEFAULT_PLOTDT = 2e-5
    DEFAULT_RUNTIME = 1.0
    DEFAULT_PLOTUPDATE_DT = 1e-2

    DEFAULT_SIMDT_KKIT = 0.1
    DEFAULT_RUNTIME_KKIT = 10.0
    DEFAULT_PLOTDT_KKIT = 1.0
    DEFAULT_PLOTUPDATE_DT_KKIT = 5.0
    
    simdt = DEFAULT_SIMDT
    plotdt = DEFAULT_PLOTDT
    runtime = DEFAULT_RUNTIME
    plotupdate_dt = DEFAULT_PLOTUPDATE_DT
    def __init__(self):
        QtCore.QObject.__init__(self)
	self._context = moose
	self._lib = moose.Neutral('/library')
	#self._proto = moose.Neutral('/proto')
	#self._data = moose.Neutral('/data')
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
        self.stopSimulation = 0
    
    def getCurrentTime(self):
        clock = moose.element('/clock')
        return clock.getField('currentTime')

        
    def getCurrentElement(self):
        return self._current_element

    def runGenesisCommand(self, cmd):
	"""Runs a GENESIS command and returns the output string"""
	self._context.runG(cmd)
        return 'In current PyMOOSE implementation running a GENESIS command does not return anything.'

    def loadModel(self, filename, filetype, target='/'):
        """Load a model from file."""
        directory = os.path.dirname(filename)
        os.chdir(directory)
        filename = os.path.basename(filename) # ideally this should not be required - but neuroML reader has a bug and gets a segmentation fault when given abosolute path.
        #moose.Property.addSimPath(directory)
        if filetype == MooseHandler.type_genesis:
            return self.loadGenesisModel(filename, target)
        elif filetype == MooseHandler.type_xml:
            return self.loadXMLModel(filename, target)
        elif filetype == MooseHandler.type_python:
            sys.path.append(directory)
            return self.loadPythonScript(filename)

    def loadGenesisModel(self, filename, target):
        """Load a model specified in a GENESIS Script.

        If the file is a kinetikit model (the criterion is 'include
        kkit' statement somewhere near the beginning of the file, it
        returns MooseHandler.type_kkit.

        Returns MooseHandler.type_genesis otherwise.
        """
        filetype = MooseHandler.type_genesis
        kkit_pattern = re.compile('include\s+kkit')
        in_comment = False
        with open(filename, 'r') as infile:
            while True:
                sentence = ''
                in_sentence = False
                line = infile.readline()
                if not line:
                    break
                line = line.strip()
                # print '#', line
                if line.find('//') == 0: # skip c++ style comments
                    # print 'c++ comment'
                    continue
                comment_start = line.find('/*')
                if comment_start >= 0:
                    in_comment = True
                    line = line[:comment_start] 
                in_sentence = line.endswith('\\')
                while in_comment and line:
                    comment_end = line.find('*/')
                    if comment_end >= 0:
                        in_comment = False
                        sentence = line[comment_end+2:] # add the rest of the line to sentence
                    line = infile.readline()
                    line = line.strip()
                while line and in_sentence:
                    sentence += line[:-1]
                    line = infile.readline()
                    if line:                    
                        line = line.strip()
                        in_sentence = line.endswith('\\')
                if line: 
                    sentence += line
                iskkit = re.search(kkit_pattern, sentence)
                # print iskkit, sentence
                if iskkit:
                    filetype = MooseHandler.type_kkit                    
                    break
        current = self._context.getCwe()
        #self._context.setCwe(target)
        self._context.loadModel(filename,target)
        self._context.setCwe(current)        
        return filetype
        
    def loadXMLModel(self, filename, target):
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
            #self._context.loadModel(filename, target)
            neuromlR = NeuroML()
            populationDict, projectionDict = neuromlR.readNeuroMLFromFile(filename)
        elif ret == MooseHandler.type_sbml:
            print 'Unsupported in GUI Mode'
        return ret    

    def loadPythonScript(self, filename):
        """Evaluate a python script."""
        extension_start = filename.rfind('.py')
        script = filename[:extension_start]
        exec 'import %s' % (script)
        #exec 'import %s as hello' % (script)
        #hello.loadGran98NeuroML_L123("Generated.net.xml")
        return MooseHandler.type_python

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
            # tableName = '%s_%d_%d' % (fieldName, self._tableSuffix, self._tableIndex)
            tableName = full_field_path[1:].replace('/', '_')
            dataNeutral = moose.Neutral(full_field_path+'/data')
            table = moose.Table(tableName, dataNeutral)
            self.fieldTableMap[full_field_path] = table
            table.stepMode = 3
            target = moose.Neutral(objPath)
            connected = table.connect('inputRequest', target, fieldName)
            config.LOGGER.info('Connected %s to %s/%s' % (table.path, target.path, fieldName))
            self._tableIndex += 1
        return table

    def updateDefaultsKKIT(self):
        MooseHandler.simdt = MooseHandler.DEFAULT_SIMDT_KKIT
        MooseHandler.plotdt = MooseHandler.DEFAULT_PLOTDT_KKIT
#        MooseHandler.simdt = (moose.element('/clock').tick)[0].dt
#        MooseHandler.plotdt = (moose.element('/clock').tick)[1].dt
        MooseHandler.plotupdate_dt = MooseHandler.DEFAULT_PLOTUPDATE_DT_KKIT
        MooseHandler.runtime = MooseHandler.DEFAULT_RUNTIME_KKIT

    def updateClocks(self, simdt, plotdt):
        moose.setClock(0, simdt)
        moose.setClock(1, simdt) 
        moose.setClock(2, simdt) 
        moose.setClock(3, simdt)
        moose.setClock(4, plotdt)

    def doReset(self, simdt, plotdt):
        """update simdt and plot dt and Reinit moose.
        """
        self.updateClocks(simdt, plotdt)
        moose.reinit()

    def doRun(self, time):
        """Just runs the simulation. 

        If time is float, it is absolute time in seconds.
        If an integer, it is the number of time steps.

        """
        #Harsha
        #continueTime helps to get the total run time required when user clicks on continue button.
        continueTime = self.getCurrentTime()+time 
        MooseHandler.runtime = time
        next_stop = MooseHandler.plotupdate_dt
        if MooseHandler.runtime < MooseHandler.plotupdate_dt:
            moose.start(MooseHandler.runtime)
            self.emit(QtCore.SIGNAL('updatePlots(float)'), self.getCurrentTime())
        else:
            while (next_stop <= MooseHandler.runtime) and (self.stopSimulation == 0):
                moose.start(MooseHandler.plotupdate_dt)
                next_stop = next_stop + MooseHandler.plotupdate_dt
                self.emit(QtCore.SIGNAL('updatePlots(float)'), self.getCurrentTime())
            if (self.getCurrentTime() < continueTime) and (self.stopSimulation == 0):
                time_left = continueTime - self.getCurrentTime()
                moose.start(time_left)
                self.emit(QtCore.SIGNAL('updatePlots(float)'), self.getCurrentTime())          
                
    def doResetAndRun(self, runtime, simdt=None, plotdt=None, plotupdate_dt=None):
        """Reset and run the simulation.

        This is to replace separate reset and run methods as two
        separate steps to run a simulation is awkward for the
        end-user.

        """
        if simdt is not None and isinstance(simdt, float): #why only float?
            MooseHandler.simdt = simdt
        if plotdt is not None and isinstance(plotdt, float):
            MooseHandler.plotdt = plotdt
        if plotupdate_dt is not None and isinstance(plotupdate_dt, float):
            MooseHandler.plotupdate_dt = plotupdate_dt
        if runtime is not None and isinstance(runtime, float):
            MooseHandler.runtime = runtime

        self.updateClocks(simdt,plotdt)
        moose.reinit()
        self.doRun(runtime)

        #if self._context.exists('/graphs'):
        #    self._context.useClock(3, '/graphs/##[TYPE=Table]')
        #if self._context.exists('/moregraphs'):
        #    self._context.useClock(3, '/moregraphs/##[TYPE=Table]')            
        #self._context.reset()


    # def doConnect(self):
    #     ret = False
    #     if self._connSrcObj and self._connDestObj and self._connSrcMsg and self._connDestMsg:
    #         ret = self._connSrcObj.connect(self._connSrcMsg, self._connDestObj, self._connDestMsg)
    #         # print 'Connected %s/%s to %s/%s: ' % (self._connSrcObj.path, self._connSrcMsg, self._connDestObj.path, self._connDestMsg), ret
    #         self._connSrcObj = None
    #         self._connDestObj = None
    #         self._connSrcMsg = None
    #         self._connDestMsg = None
    #     return ret

    # def setConnSrc(self, fieldPath):
    #     pos = fieldPath.rfind('/')
    #     moosePath = fieldPath[:pos]
    #     field = fieldPath[pos+1:]
    #     self._connSrcObj = moose.Neutral(moosePath)
    #     self._connSrcMsg = field

    # def setConnDest(self, fieldPath):
    #     pos = fieldPath.rfind('/')
    #     moosePath = fieldPath[:pos]
    #     field = fieldPath[pos+1:]
    #     self._connDestObj = moose.Neutral(moosePath)
    #     self._connDestMsg = field

    # def getSrcFields(self, mooseObj):
    #     srcFields = self._context.getFieldList(mooseObj.id, moose.FTYPE_SOURCE)
    #     sharedFields = self._context.getFieldList(mooseObj.id, moose.FTYPE_SHARED)
    #     ret = []
    #     for field in srcFields:
    #         ret.append(field)
    #     for field in sharedFields:
    #         ret.append(field)
    #     return ret

    # def getDestFields(self, mooseObj):
    #     destFields = self._context.getFieldList(mooseObj.id, moose.FTYPE_DEST)
    #     sharedFields = self._context.getFieldList(mooseObj.id, moose.FTYPE_SHARED)
    #     ret = []
    #     for field in destFields:
    #         ret.append(field)
    #     for field in sharedFields:
    #         ret.append(field)
    #     return ret
        
    def getKKitGraphs(self):
        tableList = []
        for container in moose.Neutral('/graphs').children():
            for child in moose.Neutral(container).children():
                if moose.Neutral(child).className == 'Table':
                    tableList.append(moose.Table(child))
        return tableList

    def getKKitMoreGraphs(self):
        tableList = []
        for container in moose.Neutral('/moregraphs').children():
            for child in moose.Neutral(container).children():
                if moose.Neutral(child).className == 'Table':
                    tableList.append(moose.Table(child))
        return tableList
        
    # def getDataTables(self):
    #     tableList = []
    #     for table in self._data.children():
    #         if moose.Neutral(table).className == 'Table':
    #             tableList.append(moose.Table(table))
    #     return tableList

    
    
# 
# moosehandler.py ends here

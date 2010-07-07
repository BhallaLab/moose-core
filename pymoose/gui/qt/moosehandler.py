# moosehandler.py --- 
# 
# Filename: moosehandler.py
# Description: 
# Author: subhasis ray
# Maintainer: 
# Created: Thu Jan 28 15:08:29 2010 (+0530)
# Version: 
# Last-Updated: Wed Jul  7 16:41:05 2010 (+0530)
#           By: Subhasis Ray
#     Update #: 370
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

import re
import xml.sax as sax
import xml.sax.handler as saxhandler
import xml.sax.xmlreader as saxreader
import xml.sax.saxutils as saxutils

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
        elif name == 'neuroml':
            self.model_type = MooseHandler.type_neuroml
        else:
            print name

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
    # Map between file extension and known broad filetypes.
    fileExtensionMap = {
        'Genesis Script(*.g)': type_genesis,
        'neuroML/SBML(*.xml *.bz2 *.zip *.gz)': type_xml,
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
        
        
    def runGenesisCommand(self, cmd):
	"""Runs a GENESIS command and returns the output string"""
	self._context.runG(cmd)
        return 'In current PyMOOSE implementation running a GENESIS command does not return anything.'

    def loadModel(self, filename, filetype):
        """Load a model from file."""
        directory = os.path.dirname(filename)
        os.chdir(directory)
        filename = os.path.basename(filename) # ideally this should not be required - but neuroML reader has a bug and gets a segmentation fault when given abosolute path.
        config.LOGGER.info('SIMPATH modidied to: %s' % (moose.Property.getSimPath()))
        if filetype == MooseHandler.type_genesis:
            return self.loadGenesisModel(filename)
        elif filetype == MooseHandler.type_xml:
            return self.loadXMLModel(filename)


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
        


# 
# moosehandler.py ends here

# moosehandler.py --- 
# 
# Filename: moosehandler.py
# Description: 
# Author: subhasis ray
# Maintainer: 
# Created: Thu Jan 28 15:08:29 2010 (+0530)
# Version: 
# Last-Updated: Thu Jul  1 16:20:04 2010 (+0530)
#           By: Subhasis Ray
#     Update #: 219
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
import re
import xml.sax as sax
import xml.sax.handler as saxhandler
import xml.sax.xmlreader as saxreader
import xml.sax.saxutils as saxutils

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

class MooseHandler(object):
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
    def __init__(self):
	self._context = moose.PyMooseBase.getContext()
	self._lib = moose.Neutral('/library')
	self._proto = moose.Neutral('/proto')
	self._data = moose.Neutral('/data')
        self._current_element = moose.Neutral('/')
        self._xmlreader = sax.make_parser()
        self._saxhandler = MooseXMLHandler()
        self._xmlreader.setContentHandler(self._saxhandler)
        
        
    def runGenesisCommand(self, cmd):
	"""Runs a GENESIS command and returns the output string"""
	self._context.runG(cmd)
        return 'In current PyMOOSE implementation running a GENESIS command does not return anything.'

    def loadModel(self, filename, filetype):
        """Load a model from file."""
        directory = os.path.dirname(filename)
        moose.Property.addSimPath(directory)
        os.chdir(directory)
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
            self._context.readNeuroML(filename, self._current_element.path)
        elif ret == MooseHandler.type_sbml:
            self._context.readSBML(filename, self._current_element.path)
        
        return ret    
        
        

# 
# moosehandler.py ends here

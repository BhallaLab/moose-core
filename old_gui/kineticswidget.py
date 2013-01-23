# kineticswidget.py --- 
# 
# Filename: kineticswidget.py
# Description: 
# Author: subha
# Maintainer: 
# Created: Sun Sep 23 22:27:37 2012 (+0530)
# Version: 
# Last-Updated: Mon Sep 24 01:49:32 2012 (+0530)
#           By: subha
#     Update #: 172
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

# Code:

import sys
import os
from PyQt4 import QtGui,QtCore,Qt
import pickle
import numpy as np
import random
import config
import re
import math
import collections
import kineticsutils as kutils
from kineticsutils import displayinfo
from kineticsgraphics import PoolItem, ReacItem
import moose

class KineticsDisplayWidget(QtGui.QGraphicsWidget):
    def __init__(self, modelPath, view=None):
        QtGui.QGraphicsWidget.__init__(self, view)
        # The following dicts map moose objects to corresponding
        # graphics objects. They are initialized in displayXYZ()
        # functions where Xyz can be Pool, Reactions or Enzymes.
        self.poolGraphicsMap = {}
        self.reacGraphicsMap = {}
        self.enzGraphicsMap = {}
        self.noPositionInfo = False
        # self.border = 10
        # These will be updated on reading the kkit file. These
        # present the maxmum and minimum coordinates of the objects
        # specified in the kkit file.
        self.xmax = 1.0
        self.xmin = 0.0
        self.ymax = 1.0
        self.ymin = 0.0
        # Load pickled colormap here
        cmapPath = os.path.join(
            config.settings[config.KEY_COLORMAP_DIR], 
            'rainbow2.pkl')
        cmapFile = open(cmapPath, 'rb')
        self.colorMap = pickle.load(cmapFile)
        cmapFile.close()
        self.setupDisplayInfo(modelPath)
        # This is check which version of kkit, b'cos anything below
        # kkit8 didn't had xyz co-ordinates
        if self.noPositionInfo:
            QtGui.QMessageBox.warning(self, 
                                      'No coordinates found', 
                                      'Kinetic layout works only \
for models using kkit8 or later')
            raise Exception('Unsupported kkit version')

    def xfactor(self):
        """Scale factor to translate the x position from kkit to Qt
        coordinates."""
        return self.size().width()*1.0/(self.xmax - self.xmin)

    def yfactor(self):
        """Scale factor to translate the y position from kkit to Qt
        coordinates."""
        return self.size().height()*1.0/(self.ymax - self.ymin)

    def displayReactions(self, displayDict, meshEntry):
        """Setup display of reaction obejcts"""
        ret = {}
        for reac, dinfo in displayDict.items():
            x, y = self.reposition(dinfo.x, dinfo.y)
            reacItem = ReacItem(reac, self)            
            reacItem.setDisplayProperties(
                displayinfo(x, y, dinfo.fc, dinfo.bc))
            ret[reac] = reacItem
        self.reacGraphicsMap.update(ret)
        return ret

    def displayEnzymes(self, displayDict, meshEntry):
        """Display the enzymes"""
        ret = {}
        for enzPool, dinfo in displayDict.items():
            x, y = self.reposition(dinfo.x, dinfo.y)
            enzItem = EnzItem(enzPool, self)
            enzitem.setDisplayProperties(
                displayinfo(x, y, dinfo.fc, dinfo.bc))
            ret[enzPool] = enzItem
        self.enzGraphicsMap.update(ret)
        return ret

    def displayPools(self, displayDict, meshEntry):
        ret = {}
        for pool, dinfo in displayDict.items():
            x, y = self.reposition(dinfo.x, dinfo.y)            
            poolItem = PoolItem(pool, self)        
            poolItem.setDisplayProperties(
                displayinfo(x, y, dinfo.fc, dinfo.bc))            
            ret[pool] = poolItem
        self.poolGraphicsMap.update(ret)
        return ret
        
    def displayMeshEntries(self):
        """Set up the display items for mesh entries."""
        for meshEntry, displayDict in self.meshEntryDisplay.items():
            self.displayPools(displayDict['pool'], meshEntry)
            self.displayReactions(displayDict['reaction'], meshEntry)
            self.displayEnzymes(displayDict['enzyme'], meshEntry)
        
    def setupDisplayInfo(self, modelPath):
        """Setup display information for compartments and pools and
        reactions"""
        meshEntryWildcard = modelPath + '/##[TYPE=MeshEntry]'        
        self.meshEntryDisplay = {}
        for meshEntry in moose.wildcardFind(meshEntryWildcard):
            reacDict = kutils.getReactionDisplayData(meshEntry, 
                                                     self.colorMap)
            enzDict, poolDict = kutils.getPoolDisplayData(meshEntry, 
                                                          self.colorMap)
            self.meshEntryDisplay[meshEntry] = {'reaction': reacDict, 
                                                'enzyme': enzDict,
                                                'pool': poolDict}
        xvalues = []
        yvalues = []        
        for entry in self.meshEntryDisplay.values():
            for ddict in entry.values():
                for dinfo in ddict.values():
                    xvalues.append(dinfo.x)
                    yvalues.append(dinfo.y)
        self.xmax = max(xvalues)
        self.xmin = min(xvalues)        
        self.ymax = max(yvalues)
        self.ymin = min(yvalues)
        self.noPositionInfo = len(np.nonzero(xvalues)[0]) == 0 \
            and len(np.nonzero(yvalues)[0]) == 0

    def reposition(self, x, y):
        return (self.xfactor() * (x - self.xmin), 
                self.yfactor() * (y - self.ymin))


import sys
if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    model = moose.loadModel('../Demos/Genesis_files/Kholodenko.g', 
                            '/kholodenko')
    gview = QtGui.QGraphicsView()
    scene = QtGui.QGraphicsScene(gview)
    widget = KineticsDisplayWidget(model.path)
    widget.setMinimumSize(800, 600)
    scene.addItem(widget)
    widget.displayMeshEntries()
    gview.setScene(scene)
    gview.show()
    sys.exit(app.exec_())


# 
# kineticswidget.py ends here

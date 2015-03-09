import moogli
import moose
from moose import neuroml
from PyQt4 import Qt, QtCore, QtGui
import sys
import os

app = QtGui.QApplication(sys.argv)
filename = os.path.join( os.path.split(os.path.realpath(__file__))[0]
                       , "../neuroml/CA1/CA1.morph.pop.xml"
                       )
moose.neuroml.loadNeuroML_L123(filename)
morphology = moogli.read_morphology_from_moose(name = "", path = "")
viewer = moogli.MorphologyViewerWidget(morphology)
viewer.show()
app.exec_()

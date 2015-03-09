import moogli
import moose
from moose import neuroml
from PyQt4 import Qt, QtCore, QtGui
import sys
import os

app = QtGui.QApplication(sys.argv)
filename = os.path.join( os.path.split(os.path.realpath(__file__))[0]
                       , "../neuroml/PurkinjeCellPassivePulseInput/PurkinjePassive.net.xml"
                       )
moose.neuroml.loadNeuroML_L123(filename)
morphology = moogli.read_morphology_from_moose(name = "", path = "/cells[0]")
viewer = moogli.MorphologyViewerWidget(morphology)
viewer.show()
app.exec_()

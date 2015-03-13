from PyQt4 import QtGui
from PyQt4 import QtCore
from PyQt4 import Qt
from PyQt4 import QtOpenGL


# from PyQt4.QtGui import *
from ._moogli import *
from main import *

class DynamicMorphologyViewerWidget(MorphologyViewerWidget):
    _timer = QtCore.QTimer()

    def set_callback(self,callback, idletime = 0):
        self.callback = callback
        self.idletime = idletime
        self._timer.timeout.connect(self.start_cycle)
        self.start_cycle()

    def start_cycle(self):
        if self.isVisible():
            if self.callback(self.get_morphology(), self):
                self._timer.start(self.idletime)
            else:
                self._timer.timeout.disconnect(self.start_cycle)
            self.update()
        else:
            self._timer.start(self.idletime)


__all__ = [ "Morphology"
          , "MorphologyViewer"
          , "MorphologyViewerWidget"
          , "DynamicMorphologyViewerWidget"
          ]

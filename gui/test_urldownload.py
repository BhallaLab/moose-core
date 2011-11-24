import urllib
import urllib2
from PyQt4 import QtGui, QtCore
if __name__ == '__main__':
    # @TODO for biomodels database:
    # Search POST URI:
    # http://www.ebi.ac.uk/biomodels-main/search-models.do?cmd=TEXT:SEARCH
    url = 'http://www.ebi.ac.uk/biomodels/models-main/publ/BIOMD0000000001.xml'
    request = urllib2.Request(url)
    response = urllib2.urlopen(request)
    page = response.read()
    fileout = open('biomodel.xml','w')
    fileout.write(page)
    fileout.close()
    url = 'http://www.ebi.ac.uk/biomodels'
    request = urllib2.Request(url)
    response = urllib2.urlopen(request)
    page = response.read()
    app = QtGui.QApplication([])
    mainWin = QtGui.QTextEdit()
    mainWin.setText(page)
    mainWin.show()
    app.exec_()
    print 'Finished'

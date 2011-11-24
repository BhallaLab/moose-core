# biomodelsclient.py --- 
# 
# Filename: biomodelsclient.py
# Description: 
# Author: Subhasis Ray
# Maintainer: 
# Created: Tue Mar  2 07:57:39 2010 (+0530)
# Version: 
# Last-Updated: Mon Jun  7 18:31:32 2010 (+0530)
#           By: Subhasis Ray
#     Update #: 203
# URL: 
# Keywords: 
# Compatibility: 
# 
# 

# Commentary: 
# 
# This is a client for Biomodels database SOAP service.
# It imitates the SOAP client written in JAVA
# availabele at biomodels website.

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


from suds.client import Client

BIOMODELS_WSDL_URI = 'http://www.ebi.ac.uk/biomodels-main/services/BioModelsWebServices?wsdl'
class BioModelsClient(Client):
    def __init__(self, WSDL_URI=BIOMODELS_WSDL_URI):
	"""Initialize the client with the available queries listed in
	the WSDL file. All the queries can be executed using the following syntax:
	
	client.service.queryToBeExecuted()
	"""
	try:
	    Client.__init__(self, WSDL_URI)
	except Exception, e:
	    print e

from PyQt4.Qt import Qt
from PyQt4 import QtCore, QtGui
class BioModelsClientWidget(QtGui.QWidget):
    """This is a widget with a Biomodels Client. It provides simple
    access to the biomodel queries and gives the user a view of the
    results"""
    COMBO_ITEM_QUERY_MAP = [('All Curated Model Ids', 'getAllCuratedModelsId'),
                            ('All Model Ids', 'getAllModelsId'),
                            ('All Non-curated Model Ids', 'getAllNonCuratedModelsId'),
                            ('Model Name By Id', 'getModelNameById'),
                            ('Model Ids by ChEBI', 'getModelsIdByChEBI'),
                            ('Model Ids by ChEBI Id', 'getModelsIdBuChEBIId'),
                            ('Model Ids by Gene Ontology Term', 'getModelsIdByGO'),
                            ('Model Ids by Gene Ontology Id', 'getModelsIdByGOId'),
                            ('Model Ids by Name', 'getModelsIdByName'),
                            ('Model Ids by Author/Modeler', 'getModelsIdByPerson'),
                            ('Model Ids by Publication', 'getModelsIdByPublication'),
                            ('Model Ids by Taxonomy', 'getModelsIdByTaxonomy'),]
    def __init__(self, parent=None):
	QtGui.QWidget.__init__(self, parent)
	self.client = BioModelsClient()
        self.queryPanel = QtGui.QWidget(self)
        self.queryModelLabel = QtGui.QLabel('Get ', self.queryPanel)
        self.queryModelCombo = QtGui.QComboBox(self.queryPanel)
        self.queryLineEdit = QtGui.QLineEdit(self.queryPanel)
        self.goButton = QtGui.QPushButton('Go', self.queryPanel)
        
        layout = QtGui.QHBoxLayout(self.queryPanel)
        layout.addWidget(self.queryModelLabel)
        layout.addWidget(self.queryModelCombo)
        layout.addWidget(self.queryLineEdit)
        layout.addWidget(self.goButton)
        self.queryPanel.setLayout(layout)
        for entry in BioModelsClientWidget.COMBO_ITEM_QUERY_MAP:
            self.queryModelCombo.addItem(self.tr(entry[0]), QtCore.QVariant(entry[1]))
        self.resultsPanel = QtGui.QTableWidget(self)
        layout = QtGui.QVBoxLayout(self)
        layout.addWidget(self.queryPanel)
        layout.addWidget(self.resultsPanel)
        self.setLayout(layout)
        self.setupActions()
        
	# TODO:
	# proxy = [ can be set using set_option(proxy={'http':'proxyhost:port', ...}) function

    def setupActions(self):
        self.connect(self.queryLineEdit, QtCore.SIGNAL('returnPressed()'), self.runQuery)
        self.connect(self.goButton, QtCore.SIGNAL('clicked()'), self.runQuery)

    def download(self):
        """Download the selected model"""
        selectedRow = self.resultsPanel.currentRow()
        modelId = self.resultsPanel.item(selectedRow, 0).text()
        modelSBML = self.client.service.getModelSBMLbyId(modelId)
        print modelSBML

    def runQuery(self):
        print 'Running query'
        index = self.queryModelCombo.currentIndex()
        query = self.queryModelCombo.itemData(index).toString()
        argument = self.queryLineEdit.text().trimmed()
        function = eval('self.client.service.' + str(query))
        if index > 2:
            result = function(str(argument))
        else:
            result = function()
        self.resultsPanel.clear()

        row = 0
        column = 0
        self.resultsPanel.setColumnCount(2)
        self.resultsPanel.setHorizontalHeaderItem(column, QtGui.QTableWidgetItem('Id'))
        self.resultsPanel.setHorizontalHeaderItem(column + 1, QtGui.QTableWidgetItem('Name'))
        if type(result) is type(''):
            self.resultsPanel.insertRow(row)
            item = QtGui.QTableWidgetItem(argument)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self.resultsPanel.setItem(row, column, item)
            item = QtGui.QTableWidgetItem(result)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self.resultsPanel.setItem(row, column + 1, item)
        elif type(result) is type([]):
            for value in result:
                self.resultsPanel.insertRow(row)
                item = QtGui.QTableWidgetItem(self.tr(value))
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                # item.setData(Qt.DisplayRole, QtCore.QVariant(value))
                self.resultsPanel.setItem(row, column, item)
                name = self.client.service.getModelNameById(value)
                item = QtGui.QTableWidgetItem(name)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.resultsPanel.setItem(row, column + 1, item)
                row = row + 1
        print 'Finished running query'
    
if __name__ == '__main__':
    client = BioModelsClient()
    print client.service.helloBioModels()

    app = QtGui.QApplication([])
    clientWidget = BioModelsClientWidget()
    clientWidget.show()
    app.exec_()

# 
# biomodelsclient.py ends here

# mgui.py --- 
# 
# Filename: mgui.py
# Description: 
# Author: 
# Maintainer: 
# Created: Mon Nov 12 09:38:09 2012 (+0530)
# Version: 
# Last-Updated: Thu Jul 18 10:54:33 2013 (+0530)
#           By: subha
#     Update #: 1338
# URL: 
# Keywords: 
# Compatibility: 
# 
# 

# Commentary: 
# 
# The gui driver
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
import imp
import inspect
import code
import traceback
import sys
sys.path.append('../python')
import os
from collections import defaultdict
import posixpath # We use this to create MOOSE paths
from PyQt4 import QtGui, QtCore, Qt
import config
import mplugin
import moose
import mexception
from moose import utils
from mload import loadFile
from loaderdialog import LoaderDialog
from shell import get_shell_class
from objectedit import ObjectEditDockWidget
from newmodeldialog import DialogWidget
import re
__author__ = 'Subhasis Ray'

# This maps model subtypes to corresponding plugin names. Should be
# moved to a separate property file perhaps
subtype_plugin_map = {
    'genesis/kkit': 'kkit',
    'cspace/': 'kkit'
}

    
class MWindow(QtGui.QMainWindow):
    """The main window for MOOSE GUI.

    This is the driver class that uses the mplugin API. mplugin based
    classes will provide the toolbar, plugin specific menu items and a
    set of panes to be displayed on the docks.

    1. Setting a plugin

       When a plugin is set as the current plugin, the view and the
       menus are updated.

    1.a) Updating menus: 

    the plugin can provide its own list of menus by implementing the
    function getMenus().

    the view of the plugin can also provide its own list of
    menus by implementing the function getMenus().

    the currentView provides a set of toolbars that are added to the
    main window.

    1.b) Updating views
    
    central widget is set to the currentView (a ViewBase instance) of
    the plugin.

    the currentView provides a set of panes that are inserted in the
    right dock area one by one.    

    """
    def __init__(self, *args):
        QtGui.QMainWindow.__init__(self, *args)
        self.setWindowTitle('MOOSE')
        self.pluginNames = None
        self.plugin = None
        self.fileMenu = None
        self.editMenu = None
        self.helpMenu = None
        self.helpActions = None
        self.viewActions = None
        self.editActions = None        
        self.toolBars = []
        self._loadedPlugins = {}
        self._plugins = {}
        self.setDockOptions(self.AnimatedDocks and self.AllowNestedDocks and self.AllowTabbedDocks)
        self.mdiArea = QtGui.QMdiArea()
        self.quitAction = QtGui.QAction('&Quit', self)
        self.connect(self.quitAction, QtCore.SIGNAL('triggered()'), self.quit)
        self.quitAction.setShortcut(QtGui.QApplication.translate("MainWindow", "Ctrl+Q", None, QtGui.QApplication.UnicodeUTF8))        
        self.getMyDockWidgets()       
        self.setCentralWidget(self.mdiArea)
        self.mdiArea.setViewMode(QtGui.QMdiArea.TabbedView)
        self.setPlugin('default', '/')

    def quit(self):
        QtGui.qApp.closeAllWindows()

    def handleException(self, t, v, s):
        """This handler will show warning messages for error exceptions. Show
        info at status bar for non-error exceptions. It will replace
        sys.excepthook and has the same signature (except being bound
        to this object).

        t : exception type

        v : exception value

        s: traceback object.

        """
        traceback.print_exception(t, v, s)
        title = ''.join(traceback.format_exception_only(t, v))
        trace = ''.join(traceback.format_exception(t, v, s))
        if isinstance(v, mexception.MooseInfo):
            self.statusBar().showMessage(title, 5000)
        elif isinstance(v, mexception.MooseWarning):
            QtGui.QMessageBox.warning(self, title, '\n'.join((title, trace)))
        else:
            QtGui.QMessageBox.critical(self, title, '\n'.join((title, trace)))
    
    def getPluginNames(self):
        """Return pluginNames attribute or create it by retrieving
        available plugin names from plugin/list.txt file.

        """
        if self.pluginNames is None:
            with open(os.path.join(config.MOOSE_GUI_DIR,
                                   'plugins', 
                                   'list.txt')) as lfile:
                self.pluginNames = [line.strip() for line in lfile]
                self.pluginNames = [name for name in self.pluginNames if name]
        return self.pluginNames

    def loadPluginModule(self, name, re=False):
        """Load a plugin module by name.

        First check if the plugin is already loaded. If so return the
        existing one. Otherwise, search load the plugin as a python
        module from {MOOSE_GUI_DIRECTORY}/plugins directory.

        If re is True, the plugin is reloaded.
        """
        if (not re) and name in sys.modules:
            return sys.modules[name]
        fp, pathname, description = imp.find_module(name, [config.MOOSE_PLUGIN_DIR])
        try:
            module = imp.load_module(name, fp, pathname, description)
        finally:
            if fp:
                fp.close()
        return module

    def getMyDockWidgets(self):
        """Return a list of dockwidgets that belong to the top
        level. This is needed to keep them separate from those
        provided by the plugins.

        Currently we only have shell for this."""
        if not hasattr(self, 'dockWidgets') or self.dockWidgets is None:
            self.dockWidgets = {}
            dockWidget = QtGui.QDockWidget('Python')
            dockWidget.setWidget(self.getShellWidget())
            self.dockWidgets[dockWidget] = True
            self.addDockWidget(Qt.Qt.BottomDockWidgetArea, dockWidget)
            dockWidget.setVisible(False)
            dockWidget = ObjectEditDockWidget('/')
            self.dockWidgets[dockWidget] = True
            self.objectEditDockWidget = dockWidget
            self.addDockWidget(Qt.Qt.RightDockWidgetArea, dockWidget)
            dockWidget.setVisible(False)
        return self.dockWidgets.keys()

    def getShellWidget(self):
        """Create an instance of shell widget. This can be either a
        QSciQScintialla widget or a PyCute widget (extends QTextArea)
        if the first is not available"""
        if not hasattr(self, 'shellWidget') or self.shellWidget is None:            
            self.shellWidget = get_shell_class()(code.InteractiveInterpreter(),
                                                 message='MOOSE version %s' % (moose._moose.__version__))
            self.shellWidget.interpreter.runsource('from moose import *')
            self.shellWidget.setVisible(False)
        return self.shellWidget

    def loadPluginClass(self, name, re=False):
        """Load the plugin class from a plugin module.
        
        A plugin module should have only one subclass of
        MoosePluginBase. Otherwise the first such class found will be
        loaded.
        """
        try:
            return self._loadedPlugins[name]
        except KeyError:
            pluginModule = self.loadPluginModule(name, re=re)
            for classname, classobj in inspect.getmembers(pluginModule, inspect.isclass):
                if issubclass(classobj, mplugin.MoosePluginBase):
                    self._loadedPlugins[name] = classobj
                    # classobj.getEditorView().getCentralWidget().editObject.connect(self.objectEditSlot)                        
                    return self._loadedPlugins[name]
        raise Exception('No plugin with name: %s' % (name))

    def setPlugin(self, name, root='/'):
        """Set the current plugin to use.

        This -

        1. sets the `plugin` attribute.

        2. updates menus by clearing and reinstating menus including
        anything provided by the plugin.

        3. sets the current view  to the plugins editor view.

        """
        self.plugin = self.loadPluginClass(str(name))(str(root), self)
        # try:
        #     self.plugin = self._plugins[str(name)]
        #     print 'PLUGIN', self.plugin
        #     self.plugin.setModelRoot(root)
        # except KeyError:
        #     self.plugin = self.loadPluginClass(str(name))(str(root), self)
        #     self._plugins[str(name)] = self.plugin
        self.plugin.getEditorView().getCentralWidget().editObject.connect(self.objectEditSlot, QtCore.Qt.UniqueConnection)
        self.updateMenus()
        for action in self.pluginsMenu.actions():
            if str(action.text()) == str(name):
                action.setChecked(True)
            elif action.isChecked():
                action.setChecked(False)
        for subwin in self.mdiArea.subWindowList():
            subwin.close()
        self.setCurrentView('editor')
        self.objectEditDockWidget.objectNameChanged.connect(
            self.plugin.getEditorView().getCentralWidget().updateItemSlot)
        return self.plugin

    def updateExistingMenu(self, menu):
        """Check if a menu with same title
        already exists. If so, update the same and return
        True. Otherwise return False.
        """
        if not isinstance(menu, QtGui.QMenu):
            return False
        for action in self.menuBar().actions():
            if menu.title() == action.text():
                action.menu().addSeparator()
                action.menu().addActions(menu.actions())
                return True
        return False
        
    def updateMenus(self):
        """Clear the menubar and reinstate the basic menus.  Go
        through the menus provided by current plugin and add those to
        menubar.

        If a menu provided by a plugin has same name as one of the
        core menus, the menu items provided by the plugin are appended
        to the existing menu after a separator.

        """
        self.menuBar().clear()
        menus = [self.getFileMenu(),
                 self.getEditMenu(),
                 self.getViewMenu(),
                 self.getPluginsMenu(),
                 self.getRunMenu(),
                 self.getHelpMenu()]
        for menu in menus:
            self.menuBar().addMenu(menu)
        for menu in self.plugin.getMenus():
            if not self.updateExistingMenu(menu):
                self.menuBar().addMenu(menu)

    def updateToolbars(self):
        for toolbar in self.toolBars:
            self.removeToolBar(toolbar)
        self.toolBars = []
        for toolbar in self.getMyToolBars():
            self.toolBars.append(toolbar)
        for toolbar in self.plugin.getToolBars():
            self.toolBars.append(toolbar)
        for toolbar in self.plugin.getCurrentView().getToolBars():
            self.toolBars.append(toolbar)
        for toolbar in self.toolBars:
            self.addToolBar(toolbar)
            toolbar.setVisible(True)

    def setCurrentView(self, view):
        """Set current view to a particular one: options are 'editor',
        'plot', 'run'. A plugin can provide more views if necessary.
        """
        self.plugin.setCurrentView(view)
        targetView = None
        newSubWindow = True
        widget = self.plugin.getCurrentView().getCentralWidget()
        for subwin in self.mdiArea.subWindowList():
            if subwin.widget() == widget:
                newSubWindow = False
                break
        if newSubWindow:
            subwin = self.mdiArea.addSubWindow(widget)
            subwin.setWindowTitle('%s: %s' % (view, widget.modelRoot))
            subwin.setSizePolicy(QtGui.QSizePolicy.Minimum |
                                 QtGui.QSizePolicy.Expanding, 
                                 QtGui.QSizePolicy.Minimum | 
                                 QtGui.QSizePolicy.Expanding)
            subwin.resize(600, 400)
        # Make dockwidgets from other views invisible and make those
        # from current view visible or add them if not already part of
        # main window.
        dockWidgets = set([dockWidget for dockWidget in self.findChildren(QtGui.QDockWidget)])
        for dockWidget in dockWidgets:
            if dockWidget not in self.dockWidgets:
                dockWidget.setVisible(False)
        for dockWidget in self.plugin.getCurrentView().getToolPanes():
            if dockWidget not in dockWidgets:
                self.addDockWidget(Qt.Qt.RightDockWidgetArea, dockWidget)
            else:
                dockWidget.setVisible(True)
        subwin.setVisible(True)
        self.mdiArea.setActiveSubWindow(subwin)
        self.updateMenus()
        for menu in self.plugin.getCurrentView().getMenus():
            if not self.updateExistingMenu(menu):
                self.menuBar().addMenu(menu)
        self.updateToolbars()
        return subwin

    def getMyToolBars(self):
        if not hasattr(self, 'viewToolBar'):
            self.viewToolBar = QtGui.QToolBar('View')
            self.viewToolBar.addActions(self.getViewActions())
        return [self.viewToolBar]

    def getFileMenu(self):
        if self.fileMenu is None:
            self.fileMenu = QtGui.QMenu('&File')
        else:
            self.fileMenu.clear()
        if not hasattr(self, 'newModelAction'):
            self.newModelAction = QtGui.QAction('New', self)
            self.newModelAction.setShortcut(QtGui.QApplication.translate("MainWindow", "Ctrl+N", None, QtGui.QApplication.UnicodeUTF8))
            self.connect(self.newModelAction, QtCore.SIGNAL('triggered()'), self.newModelDialogSlot)
        self.fileMenu.addAction(self.newModelAction)
        if not hasattr(self, 'loadModelAction'):
            self.loadModelAction = QtGui.QAction('L&oad model', self)
            self.loadModelAction.setShortcut(QtGui.QApplication.translate("MainWindow", "Ctrl+L", None, QtGui.QApplication.UnicodeUTF8))
            self.connect(self.loadModelAction, QtCore.SIGNAL('triggered()'), self.loadModelDialogSlot)
        self.fileMenu.addAction(self.loadModelAction)
        # self.fileMenu.addAction(self.plugin.getSaveAction())
        self.fileMenu.addAction(self.quitAction)
        return self.fileMenu

    def getEditMenu(self):
        if self.editMenu is None:
            self.editMenu = QtGui.QMenu('&Edit')
        else:
            self.editMenu.clear()
        self.editMenu.addActions(self.getEditActions())
        return self.editMenu

    def getPluginsMenu(self):
        """Populate plugins menu if it does not exist already."""
        if (not hasattr(self, 'pluginsMenu')) or (self.pluginsMenu is None):
            self.pluginsMenu = QtGui.QMenu('&Plugins')
            mapper = QtCore.QSignalMapper(self)
            pluginsGroup = QtGui.QActionGroup(self)
            pluginsGroup.setExclusive(True)
            for pluginName in self.getPluginNames():                
                action = QtGui.QAction(pluginName, self)
                action.setObjectName(pluginName)
                action.setCheckable(True)
                mapper.setMapping(action, QtCore.QString(pluginName))
                self.connect(action, QtCore.SIGNAL('triggered()'), mapper, QtCore.SLOT('map()'))
                self.pluginsMenu.addAction(action)
                pluginsGroup.addAction(action)
            self.connect(mapper, QtCore.SIGNAL('mapped(const QString &)'), self.setPlugin)
        return self.pluginsMenu

    def getHelpMenu(self):
        if self.helpMenu is None:
            self.helpMenu = QtGui.QMenu('&Help')
        else:
            self.helpMenu.clear()
        self.helpMenu.addActions(self.getHelpActions())        
        return self.helpMenu

    def getViewMenu(self):
        if (not hasattr(self, 'viewMenu')) or (self.viewMenu is None):
            self.viewMenu = QtGui.QMenu('&View')
        else:
            self.viewMenu.clear()
        self.viewMenu.addActions(self.getViewActions())
        self.docksMenu = self.viewMenu.addMenu('&Dock widgets')
        self.docksMenu.addActions(self.getDockWidgetsToggleActions())
        self.subWindowMenu = self.viewMenu.addMenu('&Subwindows')
        self.subWindowMenu.addActions(self.getSubWindowActions())
        return self.viewMenu

    # def getSubWindowVisibilityActions(self):
    #     if not hasattr(self, 'subwindowToToggles'):
    #         self.subWindowToToggle = QtGui.QSignalMapper()
    #         self.toggleToSubWindow = QtGui.QSignalMapper()
    #     for subwindow in self.mdiArea.subWindowList():
    #         if self.subWindowToToggle.mapping(subwindow) is None:
    #             action = QtGui.QAction('Show: %s' % (subwindow.windowTitle), self)                
    #             self.toggleToSubWindow.setMapping(action, subwindow)
    #             self.connect(action, QtCore.SIGNAL('triggered()'),
    #                          self.toggleToSubWindow,
    #                          QtCore.SLOT('mapped(QWidget*)'))
    #             self.subWindowToToggle.setMapping(subwindow, action)
    #             self.connect(subwindow, QtCore.SIGNAL('closed()')
            
    #     self.subWindowVisibilityMenu = QtGui.Q
    #     for subwin in self.mdiArea.subWindowList():
            

    def getRunMenu(self):
        if (not hasattr(self, 'runMenu')) or (self.runMenu is None):
            self.runMenu = QtGui.QMenu('&Run')
        else:
            self.runMenu.clear()
        self.runMenu.addActions(self.getRunActions())
        return self.runMenu
    
    def getEditActions(self):
        if (not hasattr(self, 'editActions')) or (self.editActions is None):
            self.setModelRootAction = QtGui.QAction('&Set model root', self)
            self.setModelRootAction.triggered.connect(self.showSetModelRootDialog)
            self.setDataRootAction = QtGui.QAction('Set &data root', self)
            self.setDataRootAction.triggered.connect(self.showSetDataRootDialog)
            self.editActions = [self.setModelRootAction, self.setDataRootAction]
        return self.editActions

    def showSetModelRootDialog(self):
        root, ok = QtGui.QInputDialog.getText(self, 'Model Root', 'Enter the model root path:', text=moose.element(self.plugin.modelRoot).path)
        if not ok:
            return
        root = str(root) #convert from QString to python str
        self.plugin.setModelRoot(root)
        for subwin in self.mdiArea.subWindowList():
            if hasattr(subwin.widget(), 'modelRoot'):
                subwin.setWindowTitle(root)

    def showSetDataRootDialog(self):
        root, ok = QtGui.QInputDialog.getText(self, 'Data Root', 'Enter the data root path:', text=moose.element(self.plugin.dataRoot).path)
        if not ok:
            return
        root = str(root) #convert from QString to python str
        self.plugin.setDataRoot(root)
        for subwin in self.mdiArea.subWindowList():
            if hasattr(subwin.widget(), 'dataRoot'):
                subwin.setWindowTitle(root)

    def getViewActions(self):
        if (not hasattr(self, 'viewActions')) or (self.viewActions is None):
            self.editorViewAction = QtGui.QAction('&Editor view', self)
            self.editorViewAction.triggered.connect(self.openEditorView)
            self.plotViewAction = QtGui.QAction('&Plot view', self)
            self.plotViewAction.triggered.connect(self.openPlotView)
            self.runViewAction = QtGui.QAction('&Run view', self)
            self.runViewAction.triggered.connect(self.openRunView)     
            self.viewActions = [self.editorViewAction, self.plotViewAction, self.runViewAction]
        return self.viewActions

    def getSubWindowActions(self):
        if not hasattr(self, 'subWindowActions') or self.subWindowActions is None:
            self.tabbedViewAction = QtGui.QAction('&Tabbed view', self)
            self.tabbedViewAction.triggered.connect(lambda : self.mdiArea.setViewMode(QtGui.QMdiArea.TabbedView))
            self.subWindowViewAction = QtGui.QAction('&SubWindow view', self)
            self.subWindowViewAction.triggered.connect(lambda : self.mdiArea.setViewMode(QtGui.QMdiArea.SubWindowView))
            self.tileSubWindowsAction = QtGui.QAction('Ti&le subwindows', self)
            self.tileSubWindowsAction.triggered.connect(self.mdiArea.tileSubWindows)
            self.cascadeSubWindowsAction = QtGui.QAction('&Cascade subwindows', self)
            self.cascadeSubWindowsAction.triggered.connect(self.mdiArea.cascadeSubWindows)
            self.subWindowActions = [self.tabbedViewAction, self.subWindowViewAction, self.tileSubWindowsAction, self.cascadeSubWindowsAction]
        return self.subWindowActions

    def getDockWidgetsToggleActions(self):
        """Get a list of actions for toggling visibility of dock
        widgets

        """
        return [widget.toggleViewAction() for widget in self.findChildren(QtGui.QDockWidget)]

    def getRunActions(self):
        if (not hasattr(self, 'runActions')) or \
                (self.runActions is None):
            self.startAction = QtGui.QAction('Start', self)
            self.startAction.triggered.connect(self.resetAndStartSimulation)
            self.pauseAction = QtGui.QAction('Pause', self)
            self.pauseAction.triggered.connect(self.pauseSimulation)
            self.continueAction = QtGui.QAction('Continue', self)
            self.continueAction.triggered.connect(self.continueSimulation)
            self.runActions = [self.startAction, self.pauseAction, self.continueAction]
        return self.runActions

    def getHelpActions(self):
        if (not hasattr(self, 'helpActions')) or (self.helpActions is None):
            self.actionAbout = QtGui.QAction('About MOOSE', self)
            self.connect(self.actionAbout, QtCore.SIGNAL('triggered()'), self.showAboutMoose)
            self.actionBuiltInDocumentation = QtGui.QAction('Built-in documentation', self)
            self.connect(self.actionBuiltInDocumentation, QtCore.SIGNAL('triggered()'), self.showBuiltInDocumentation)
            self.actionBug = QtGui.QAction('Report a bug', self)
            self.connect(self.actionBug, QtCore.SIGNAL('triggered()'), self.reportBug)
            self.helpActions = [self.actionAbout, self.actionBuiltInDocumentation, self.actionBug]
        return self.helpActions
        
    def showAboutMoose(self):
        with open(config.MOOSE_ABOUT_FILE, 'r') as aboutfile:
            QtGui.QMessageBox.about(self, 'About MOOSE', ''.join(aboutfile.readlines()))

    def showDocumentation(self, source):
        if not hasattr(self, 'documentationViewer'):
            self.documentationViewer = QtGui.QTextBrowser()
            self.documentationViewer.setOpenLinks(True)
            self.documentationViewer.setOpenExternalLinks(True)
            self.documentationViewer.setSearchPaths([config.settings[config.KEY_DOCS_DIR],
                                                     os.path.join(config.settings[config.KEY_DOCS_DIR], 'html'),
                                                     os.path.join(config.settings[config.KEY_DOCS_DIR], 'images')])
            self.documentationViewer.setMinimumSize(800, 480)
        self.documentationViewer.setSource(QtCore.QUrl(source))
        result = self.documentationViewer.loadResource(QtGui.QTextDocument.HtmlResource, self.documentationViewer.source())
        if not result.isValid():
            QtGui.QMessageBox.warning(self, 'Could not access documentation', 'The link %s could not be accessed' % (source))
            return
        self.documentationViewer.setWindowTitle(source)
        self.documentationViewer.reload()
        self.documentationViewer.setVisible(True)

    def reportBug(self):
        QtGui.QDesktopServices.openUrl(QtCore.QUrl(config.MOOSE_REPORT_BUG_URL))

    def showBuiltInDocumentation(self):
        self.showDocumentation('moosebuiltindocs.html')

    # openEditorView, openPlotView and openRunView are identical
    # except the view they ask from the plugin. Consider using a
    # mapper.
    def openEditorView(self):
        """Switch to the editor view of current plugin. If there is
        already a subwindow for this, make that the active
        one. Otherwise create a new one.

        """
        self.setCurrentView('editor')

    def openPlotView(self):
        self.setCurrentView('plot')
        
    def openRunView(self):
        self.setCurrentView('run')

    def resetAndStartSimulation(self):
        """TODO this should provide a clean scheduling through all kinds
        of simulation or default scheduling should be implemented in MOOSE
        itself. We need to define a policy for handling scheduling. It can
        be pushed to the plugin-developers who should have knowledge of
        the scheduling criteria for their domain."""
        settings = config.MooseSetting()
        try:
            simdt_kinetics = float(settings[config.KEY_KINETICS_SIMDT])
        except ValueError:
            simdt_kinetics = 0.1
        try:
            simdt_electrical = float(settings[config.KEY_ELECTRICAL_SIMDT])
        except ValueError:
            simdt_electrical = 0.25e-4
        try:
            plotdt_kinetics = float(settings[config.KEY_KINETICS_PLOTDT])
        except ValueError:
            plotdt_kinetics = 0.1
        try:
            plotdt_electrical = float(settings[config.KEY_ELECTRICAL_PLOTDT])
        except ValueError:
            plotdt_electrical = 0.25e-3
        try:
            simtime = float(settings[config.KEY_SIMTIME])
        except ValueError:
            simtime = 1.0
        moose.reinit()
        view = self.plugin.getRunView()
        moose.start(simtime)
        if view.getCentralWidget().plotAll:
            view.getCentralWidget().plotAllData()
        self.setCurrentView('run')        

    def pauseSimulation(self):
        moose.stop()

    def continueSimulation(self):
        """TODO implement this somewhere else"""
        try:
            simtime = float(config.MooseSetting()[config.KEY_SIMTIME])
        except ValueError:
            simtime = 1.0
        moose.start(simtime)      

    #Harsha: added visible=True so that loadModelDialogSlot and NewModelDialogSlot call this function
    #        to clear out object path
    def objectEditSlot(self, mobj, visible=True):
        """Slot for switching the current object in object editor."""
        self.objectEditDockWidget.setObject(mobj)        
        self.objectEditDockWidget.setVisible(visible)

    def loadModelDialogSlot(self):
        """Start a file dialog to choose a model file.

        Once the dialog succeeds, we should hand-over the duty of
        actual model loading to something else. Then refresh the
        views. Things to check from the user:

        1) The file type

        2) Target element

        3) Whether we should update the current window or start a new
        window.

        4) Plugin to use for displaying this model (can be automated
        by looking into the model file for a regular expression)

        """
        activeWindow = None # This to be used later to refresh the current widget with newly loaded model
        dialog = LoaderDialog(self, 
                              self.tr('Load model from file'))
        
        if dialog.exec_():
            fileNames = dialog.selectedFiles()
            for fileName in fileNames:
                # print 'Current plugin', self.plugin
                modelName = dialog.getTargetPath()
                if '/' in modelName:
                    raise mexception.ElementNameError('Model name cannot contain `/`')
                ret = loadFile(str(fileName), '/model/%s' % (modelName), merge=False)
                #print '11111'
                # Harsha: if subtype is None, in case of cspace then pluginLookup = /cspace/None 
                #     which will not call kkit plugin so cleaning to /cspace 
                pluginLookup = '%s/%s' % (ret['modeltype'], ret['subtype'])
                try:
                    pluginName = subtype_plugin_map['%s/%s' % (ret['modeltype'], ret['subtype'])]
                except KeyError:
                    pluginName = 'default'
                print 'Loaded model', ret['model'].path
                self.setPlugin(pluginName, ret['model'].path)
                #Harsha: This will clear out object editor's objectpath and make it invisible
                self.objectEditSlot('/',False)

    def newModelDialogSlot(self):
        #Harsha: Create a new dialog widget for model building
        newModelDialog = DialogWidget()
        if newModelDialog.exec_():
            modelPath = str(newModelDialog.modelPathEdit.text()).strip()
            if len(modelPath) == 0:
                raise mexception.ElementNameError('Model path cannot be empty')
            if re.search('[ /]',modelPath) is not None:
                raise mexception.ElementNameError('Model path should not containe / or whitespace')
            plugin = str(newModelDialog.submenu.currentText())
            #Harsha: All model will be forced to load/build under /model, 
            modelContainer = moose.Neutral('/model')
            modelRoot = moose.Neutral('%s/%s' % (modelContainer.path, modelPath))
            self.setPlugin(plugin, modelRoot.path)
            #Harsha: This will clear out object editor's objectpath and make it invisible
            self.objectEditSlot('/', False)


if __name__ == '__main__':
    # create the GUI application
    app = QtGui.QApplication(sys.argv)
    QtGui.qApp = app
    icon = QtGui.QIcon(os.path.join(config.KEY_ICON_DIR,'moose_icon.png'))
    app.setWindowIcon(icon)
    # instantiate the main window
    #moose.loadModel('../Demos/Genesis_files/Kholodenko.g','/kho')
    mWindow =  MWindow()
    mWindow.setWindowState(QtCore.Qt.WindowMaximized)
    sys.excepthook = mWindow.handleException
    # show it
    mWindow.show()
    # start the Qt main loop execution, exiting from this script
    #http://code.google.com/p/subplot/source/browse/branches/mzViewer/PyMZViewer/mpl_custom_widget.py
    #http://eli.thegreenplace.net/files/prog_code/qt_mpl_bars.py.txt
    #http://lionel.textmalaysia.com/a-simple-tutorial-on-gui-programming-using-qt-designer-with-pyqt4.html
    #http://www.mail-archive.com/matplotlib-users@lists.sourceforge.net/msg13241.html
    # with the same return code of Qt application
    config.settings[config.KEY_FIRSTTIME] = 'False' # string not boolean
    sys.exit(app.exec_())
    

# 
# mgui.py ends here

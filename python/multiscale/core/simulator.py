#!/usr/bin/env python 

"""simulator.py:  This class reads the variables needed for simulation and
prepare moose for simulation.

Last modified: Tue Feb 18, 2014  07:23PM

"""

__author__           = "Dilawar Singh"
__copyright__        = "Copyright 2013, NCBS Bangalore"

__credits__          = ["NCBS Bangalore", "Bhalla Lab"]
__license__          = "GPL"
__version__          = "1.0.0"
__maintainer__       = "Dilawar Singh"
__email__            = "dilawars@iitb.ac.in"
__status__           = "Development"

import moose 
import debug.debug as debug
import inspect
import numpy
import pylab
import os
import logging 
import core.config as config
import numpy as np
import logging
import helper.moose_methods as moose_methods
import collections
import re
import sys

class Variable:
    '''Variable to plot. 
    '''
    def __init__(self):
        ''' Default constructor '''
        self.name = None
        self.plotPath = None
        self.type = None
        self.namespace = None
        self.moosePath = None
        self.xml = None

    def __init__(self, name, type):
        '''Another constructor '''
        self.name = name
        self.type = type
    

class Simulator:
    '''Responsible for all simulation related activity. 
    '''

    def __init__(self, arg):
        '''
        Simulator class
        '''
        self.arg = arg
        self.simXml = arg[0]
        self.simXmlPath = arg[1]
        self.simElemName = "element"
        self.cellPath = config.cellPath
        self.libraryPath = config.libraryPath
        self.rootElem = self.simXml.getroot()
        self.variablesToPlot = list()
        self.tableClockId = 3
        self.tablePath = '/Tables'
        moose.Neutral(self.tablePath)

        # For each file, a list of plots.
        self.plotFiles = collections.defaultdict(set)

        self.elecPath = self.rootElem.get('elec_path', '/neuroml/electrical')
        if not moose.exists(self.elecPath):
            debug.printDebug("USER", 
                    [ "Path {} does not exists in mooose.".format(self.elecPath)
                        , "You sure this is a valid elec_path in config.xml"
                        ]
                    , frame = inspect.currentframe()
                    )
            debug.printDebug("USER", "Not simulating..")
            raise UserWarning("Missing moose path {}".format(self.elecPath))

        self.globalVar = self.rootElem.find('global')
        self.simDt = float(self.globalVar.get('sim_dt', '1e-3'))
        self.plotDt = float(self.globalVar.get('plot_dt', '1e-3'))
        self.simMethod = self.globalVar.get('sim_method', 'ee')
        self.simulate = True
        self.totalPlots = 0
        self.currentCompartmentPath = ''

        self.figurePath = self.rootElem.get('save_path', '.')
        if not os.path.isdir(self.figurePath):
            os.makedirs(self.figurePath)

        if self.rootElem.get('simulate') == "false":
            self.simulate = False
        self.runtime = float(self.rootElem.get('runtime', '10.0'))

        # this dictionary keeps the object created by function setupTable in a
        # variable name. This object of setupTable can be accessed later 
        self.tableDict = {}

    def updateMoose(self, populationDict, projectionDict):
        """Update the moose, as per the simulation specific settings.
        """
        self.popDict = populationDict
        self.projDict = projectionDict

        # Add simulation element e
        [self.setupRecord(x) for x in self.simXml.findall(self.simElemName)]
        
        ## Start simulation.
        try:
            self.simulateNetwork()
        except Exception as e:
            raise e
        self.plot(plotInASCII=True)

    def setupRecord(self, xmlElem):
        '''Setup record.
        '''
        elemType = xmlElem.get('type', 'soma')
        if elemType == 'soma':
            # Now get all records and setup moose for them.
            recordXml = xmlElem.find('record')
            try:
                self.setupSomaRecord(recordXml, xmlElem.attrib)
            except Exception as e:
                debug.printDebug("WARN"
                        , "Failed to setup record on soma"
                        , frame = inspect.currentframe()
                        )
                raise e
        else:
            debug.printDebug("TODO"
                    , "This type is not implemented yet %s" % elemType 
                    )
            return

    def setupSomaRecord(self, recordXml, params):
        """
        Params is a dictionary having attributes to <element>
        """
        debug.printDebug("STEP", "Setting up records on soma")

        [self.setupRecordsInMoose(v, params, recordXml.attrib) for 
                v in recordXml.findall('variable')]

        # All tables updates at tableclock
        moose.useClock(self.tableClockId, self.tablePath+'/##', 'process')


    def getPath(self, populationType, instanceId):
        ''' Given a population type and instanceId, return its path '''
        try:
            path = self.popDict[populationType][1]
        except KeyError as e:
            debug.printDebug("ERROR"
                    , [ "Population type `{0}` not found".format(populationType)
                        , "Availale population in network are "
                        , self.popDict.keys()
                    ]
                    )
            raise KeyError("Missing population type : {}".format(populationType))
        except Exception as e:
            raise e

        try:
            path = path[instanceId]
        except KeyError as e:
            msg = "Population type {} has no instance {}".format(
                    populationType
                    , instanceId
                    )
            debug.printDebug("ERROR"
                    , [msg , "Available instances are" , path.keys() ]
                    )
            raise KeyError(msg)

        # Now get the path from moose path
        path = path.path
        if not re.match(r'(\/\w+)+', path):
            raise UserWarning("{} is not a valid path".format(path))
        return path 


    def setupRecordsInMoose(self, variable, params, recordDict):
        '''
        Add a variable to plot.
        '''
        populationType = params['population']
        variableType = params['type']
        instanceId = int(params['instance_id'])
        
        # Not all types of exceptions are raised by python bindings. When c++
        # failes, it should print more details messages.
        targetBasePath = self.getPath(populationType, instanceId)

        varName = variable.get('name')
        target = variable.find("target_in_simulator")
        targetType = target.get('type')
        rootPath = target.get('path')
        # If the path is not prefixed by element then take the absolute path.
        if target.get('prefixed_by_element', 'true') == "false":
            path = rootPath
        else: 
            path = os.path.join(targetBasePath, rootPath)

        # If outputFile inside the record element is None then use the global
        # name, if this is also none, then use the default gui to display the
        # model.
        outputFileName = variable.get('output'
                , recordDict.get('file_name', None)
                )

        print "here"
        var = Variable(varName, variableType)
        var.moosePath = path
        var.xml = variable 
        if outputFileName:
            self.plotFiles[outputFileName].add(var)
        else:
            self.plotFiles['matplotlib_gui'].add(var)

        # Path has been set, now attach it to moooooose.
        path = path.strip()
        targetVariable = variableType + varName
        debug.printDebug("DEBUG"
                , "Record table: {0} <= {1}".format(targetVariable, path)
                )

        if targetType == "Compartment":
            self.tableDict[targetVariable] = (moose_methods.setupTable(
                    targetVariable
                    , moose.Compartment(path)
                    , varName
                    , self.tablePath
                    ), path)
        elif targetType == "CaConc":
            self.tableDict[targetVariable] = (moose_methods.setupTable(
                    targetVariable
                    , moose.CaConc(path)
                    , varName
                    , self.tablePath
                    ), path)
        elif targetType == "HHChannel":
            self.tableDict[targetVariable] = (moose_methods.setupTable(
                    targetVariable
                    , moose.HHChannel(path)
                    , varName
                    , self.tablePath
                    ), path)
        else:
            debug.printDebug("WARN"
                    , "Unsupported type {0}".format(targetVariable)
                    , frame = inspect.currentframe()
                    )
        
    def simulateNetwork(self):
        ''' 
        Start simulation.
        '''
        # Now reinitialize moose
        assert self.simDt > 0.0
        assert self.plotDt > 0.0 

        debug.printDebug("STEP"
                , "Simulate with simDt({0}), plotDt({1}), simMethod({2})".format(
                    self.simDt
                    , self.plotDt
                    , self.simMethod
                    )
                )
        try:
            moose.utils.resetSim([self.elecPath, self.cellPath]
                    , self.simDt
                    , self.plotDt
                    , simmethod = self.simMethod 
                    )

        except Exception as e:
            raise e
        if self.simulate:
            debug.printDebug("DEBUG"
                    , "Simulating for {0} seconds, Using methods {1}".format(
                            self.runtime
                            , self.simMethod
                            )
                    )
            try:
                moose.reinit()
            except Exception as e:
                raise e
            try:
                moose.start(self.runtime)
            except Exception as e:
                raise e
        else:
            debug.printDebug("WARN"
                    , "Simulation is set to false. Nothing to do"
                    )
            sys.exit()


    def plot(self, plotInASCII=False):
        # After simulation, plot user-requested variables.
        
        # Now we have a dictionary in which for each key there is set of plot
        # variables. Configure these plots.
        for filename in self.plotFiles:
            self.plotVars(filename, self.plotFiles[filename], plotInASCII)
            if not plotInASCII:
                if filename != "matplotlib_gui":
                    debug.printDebug("INFO"
                            , "Saving to file {}".format(filename)
                            )
                    pylab.savefig(filename)
                else:
                    pylab.show()
        
    def plotVars(self, filename, setOfVariables, plotInASCII):
        ''' Plot these records on one file '''
        self.totalPlots = len(setOfVariables)
        for i, var in enumerate(setOfVariables):
            debug.printDebug("DEBUG"
                    , "Ploting %s, %s " % (var.name, var.moosePath)
                    )
            variableXml = var.xml
            plotXml = variableXml.find('plot')

            configDict = self.configurePlot(plotXml)
            if not configDict.get('title'):
                configDict['title'] = var.moosePath
            yvec, xvec = self.getPlotData(var.type+var.name)
            pylab.subplot(self.totalPlots, 1, i)
            self.plotVar(xvec, yvec, configDict, plotInASCII)


    def plotVar(self, xvec, yvec, plotParams, plotInASCII=False):
        '''Plot a variable. '''
        if plotInASCII:
            self.plotInTerminal(xvec, yvec, plotParams)
        else:
            plotArgs = plotParams.get('plot_args', '')
            try:
                pylab.plot(xvec, yvec, plotArgs)
            except Exception as e:
                debug.printDebug("WARN"
                        , "Failed to plot with error")
                raise e

            xlabel = plotParams.get('xlabel', None)
            pylab.xlabel(plotParams.get('xlabel', ''), fontsize=8)
            pylab.ylabel(plotParams.get('ylabel', ''), fontsize=8)
            pylab.title(plotParams.get('title', ''), fontsize=8)
            pylab.tick_params(labelsize='6')
            
        return None

    def plotInTerminal(self, xvec, yvec, plotParams):
        '''
        Plot given vectors in terminal
        '''
        import subprocess
        g = subprocess.Popen(["gnuplot"], stdin=subprocess.PIPE)
        g.stdin.write("set term dumb 79 25\n")
        g.stdin.write("plot '-' using 1:2 title 'Line1' with linespoints \n")
        for i,j in zip(xvec, yvec):
            g.stdin.write("%f %f\n" % (i, j))
        g.stdin.write("e\n")
        g.stdin.flush()


    def configurePlot(self, plotXml):
        '''
        Configure plot according to plotXml
        '''
        plotType = plotXml.get('type', '2d')
        if plotType == "2d":
            return self.configure2DPlot(plotXml)
        else:
            logging.warning("Plot type %s is not supported yet." % plotType)
            debug.printDebug("TODO"
                    , [ "Plot type %s is not supported yet " % plotType
                        , "Existing without doing anything ... "
                        ]
                    )
            return None
        

    def configure2DPlot(self, plotXml):
        """
        Configure 2D plots
        """

        plotParams = dict()
        xXml = plotXml.find('x')
        yXml = plotXml.find('y')
        try:
            self.start = float(xXml.attrib['start'])
        except Exception as e:
            self.start = 0.0
        assert self.start >= 0.0, "Must start at >= 0.0 time. Right?"

        try:
            self.step = float(xXml.attrib['step'])
        except Exception as e:
            try:
                self.step = self.simDt 
            except Exception as e:
                self.step = 10e-6
                debug.printDebug("WARN"
                        , "No step size specified. Using {0}".format(self.step)
                        )
        if self.step <= 0.0:
            raise RuntimeError("Zero or negative step size")
            
        try:
            self.stop = float(xXml.attrib['stop'])
        except Exception as e:
            self.stop = self.runtime
            logging.debug(
                    "No stop time is specified. Using default %s" % self.stop
                    )
        assert self.stop > 0.0
        moose.setClock(self.tableClockId, self.step)

        # Label and title
        labelXml = plotXml.find('label')
        plotParams['output_file'] = plotXml.get('output', None)
        if labelXml is not None:
            plotParams['xlabel'] = labelXml.get('x')
            plotParams['ylabel'] = labelXml.get('y')
            plotParams['title'] = labelXml.get('title', None)

        return plotParams

    def getPlotData(self, varName):
        """Get the plot data from dictionary
        """
        assert varName in self.tableDict.keys(), "{} not found in {}".format(
                varName
                , self.tableDict.keys()
                )
        
        tableObj, self.currentCompartmentPath = self.tableDict[varName]
        
        # If there is nothing to plot, quit
        if len(tableObj.vector) < 1:
            debug.printDebug("WARN", "Empty data-set. Nothing to plot.")
            return None, None

        xvec = numpy.arange(self.start, self.stop*10, self.step)

        # Resize xvec to fit the simulation size.
        logging.debug(
                "Resizing vector of {} to fit in simulation size {}".format(
                    varName
                    , tableObj.vec.size
                    )
                )
        xvec = xvec[ : tableObj.vector.size ]
        return tableObj.vector, xvec


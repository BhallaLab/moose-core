# This is our main script.
import os
import sys
import debug.debug as debug
import inspect
import parser.NeuroML as NeuroML
import core.mumbl as mumbl
import core.simulator as moose_config
import moose
from IPython import embed
from debug.logger import *
import helper.moose_methods as mm


from lxml import etree

def ifPathsAreValid(paths) :
    ''' Verify if path exists and are readable. '''
    if paths :
        paths = vars(paths)
        for p in paths :
            if not paths[p] : continue
            for path in paths[p] :
                if not path : continue
                if os.path.isfile(path) : pass
                else :
                    debug.printDebug("ERROR"
                        , "Filepath {0} does not exists".format(path))
                    return False
            # check if file is readable
            if not os.access(path, os.R_OK) :
              debug.printDebug("ERROR", "File {0} is not readable".format(path))
              return False
    return True

# standard module for building a command line parser.
import argparse

# This section build the command line parser
argParser = argparse.ArgumentParser(description= 'Mutiscale modelling of neurons')
argParser.add_argument('--nml', metavar='nmlpath'
    , required = True
    , nargs = '+'
    , help = 'nueroml model'
    )
argParser.add_argument('--mumbl', metavar='mumbl', required = True, nargs = '+'
    , help = 'Lanaguge to do multi-scale modelling in moose'
    )
argParser.add_argument('--config', metavar='config', required = True
    , nargs = '+'
    , help = 'Simulation specific settings for moose in a xml file'
    )
args = argParser.parse_args()

import parser.parser as parser

if args:
    if ifPathsAreValid(args):
        debug.printDebug("INFO", "Started parsing XML models")
        etreeDict = parser.parseXMLs(args, validate=False)
        debug.printDebug("INFO", "Parsing of XMLs is done")

        nml = etreeDict['nml'][0]
        nmlObj = NeuroML.NeuroML()
        populationDict, projectionDict = nmlObj.loadNML(nml)

        # Start processing mumbl
        mumblObj = mumbl.Mumble(etreeDict['mumbl'][0])
        mumblObj.load()

        debug.printDebug("STEP", "Updating moose for simulation")
        simObj = moose_config.Simulator(etreeDict['config'][0])

        simObj.updateMoose(populationDict, projectionDict)

        try:
            mm.writeGraphviz(filename="./figs/topology.dot"
                    , filterList = ["classes", "Msgs", "clock"]
                    )
        except Exception as e:
            debug.printDebug("ERROR"
                    , "Failed to write a graphviz file: %s " % e
                    )
    else:
        debug.printDebug("FATAL", "One or more model file does not exists.")
        sys.exit()
else:
    debug.printDebug("FATAL", "Please provide at least one model. None given.")
    sys.exit()

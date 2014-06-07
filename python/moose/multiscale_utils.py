# This file is part of MOOSE simulator: http://moose.ncbs.res.in.

# MOOSE is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# MOOSE is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with Foobar.  If not, see <http://www.gnu.org/licenses/>.


"""multiscale_utils.py: 

    Utilities related to multi_scale modeling.

Last modified: Sat Jan 18, 2014  05:01PM

"""
    
__author__           = "Dilawar Singh"
__copyright__        = "Copyright 2013, NCBS Bangalore"
__credits__          = ["NCBS Bangalore", "Bhalla Lab"]
__license__          = "GNU GPL"
__version__          = "1.0.0"
__maintainer__       = "Dilawar Singh"
__email__            = "dilawars@ncbs.res.in"
__status__           = "Development"


import _moose 
import inspect
import os
import sys
import print_utils
import helper.moose_methods as moose_methods
import inspect
import mumble

# Function to call mumble 
def loadMumble( mumbleFile):
    """Load mumble into mooose """
    if not os.path.isfile(mumbleFile):
        print_utils.dump("ERROR"
                , "File %s does not exists or unreadable" % mumbleFile
                )
        sys.exit(0)
    mumble = mumble.Mumble()

if __name__ == '__main__':
    import parser.NeuroML as NeuroML
    import helper.simulator as simulator
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
        

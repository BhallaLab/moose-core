#!/usr/bin/env python

"""sim_utils.py: 
    Helper function related with simulation.

Last modified: Thu Jun 12, 2014  02:44PM

"""
    
__author__           = "Dilawar Singh"
__copyright__        = "Copyright 2013, NCBS Bangalore"
__credits__          = ["NCBS Bangalore", "Bhalla Lab"]
__license__          = "GPL"
__version__          = "1.0.0"
__maintainer__       = "Dilawar Singh"
__email__            = "dilawars@iitb.ac.in"
__status__           = "Development"

import _moose
import os
import re
import inspect
import print_utils
import verification_utils 

##
# @brief Run the simulation in moose. Run various verification tests before
# running the simulation.
#
# @param runTime. Float. Time to run the simulation.
# @param verify. Bool, default True. Run some tests before running simulation.
#
# @return None

def run(runTime, verify=True):
    """Run the simulation for runTime """
    if not verify:
        _moose.run(runTime)
        return 
    try:
        verification_utils.verify()
    except Exeption as e:
        print_utils.dump("INFO", "Failed to simulate with error %s " % e)
        sys.exit(0)
    print_utils.dump("INFO", "Running simulation for {} sec".format(runTime))
    _moose.start(runTime)


def recordTarget(tablePath, target, field = 'vm', **kwargs):
    """Setup a table to record at given path.

    Make sure that all root paths in tablePath exists.

    Returns a table.
    """

    # If target is not an moose object but a string representing intended path
    # then we need to fetch the object first.

    if type( target) == str:
        if not _moose.exists(target):
            msg = "Given target `{}` does not exists. ".format( target )
            raise RuntimeError( msg )
        else:
            target = _moose.Neutral( target )

    assert target.path, "Target must have a valid moose path."

    table = _moose.Table( tablePath )
    assert table

    # Sanities field. 
    if field == "output":
        pass
    elif 'get' not in field:
        field = 'get'+field[0].upper()+field[1:]
    else:
        field = field[:2]+field[3].upper()+field[4:]
    try:
        print_utils.dump("TABLE"
                , "Connecting table {} to target {} field {}".format(
                    table.path
                    , target.path
                    , field
                    )
                )
        table.connect( 'requestOut', target, field )
    except Exception as e:
        print_utils.dump("ERROR"
                , [ "Failed to connect table to target"
                    , e
                    ]
                )
        raise e
    assert table, "Moose is not able to create a recording table"
    return table

 
nameSep = '_'

def toFloat(string):
    if type(string) == float:
        return string 
    elif type(string) == str:
        return stringToFloat(string)
    else:
        raise RuntimeError("Converting type %s to float" % type(str))

def commonPath(pathA, pathB):
    ''' Find common path at the beginning of two paths. '''
    a = pathA.split('/')
    b = pathB.split('/')
    common = []
    for (i, p) in enumerate(a):
        if a[i] == b[i]: 
            common.append(p)
        else: 
            return '/'.join(common)
    return '/'.join(common)

def ifPathsAreValid(paths):
    ''' Verify if path exists and are readable. '''
    if not paths:
        return False
    paths = vars(paths)
    for p in paths:
        if not paths[p] : continue
        for path in paths[p] :
            if not path : continue
            if os.path.isfile(path) : pass
            else :
                print_utils.dump("ERROR"
                        , "Filepath {0} does not exists".format(path))
                return False
        # check if file is readable
        if not os.access(path, os.R_OK):
            print_utils.dump("ERROR", "File {0} is not readable".format(path))
            return False
    return True


def moosePath(baseName, append):
    """ 
    Append instance index to basename

    """
    if append.isdigit():
        if len(nameSep) == 1:
            return baseName + nameSep + append 
        elif len(nameSep) == 2:
            return baseName + nameSep[0] + append + nameSep[1]
        else:
            raise UserWarning, "Not more than 2 characters are not supported"
    else:
        return "{}/{}".format(baseName, append)


def splitComparmentExpr(expr):
    """ Breaks compartment expression into name and id.
    """
    if len(nameSep) == 1:
        p = re.compile(r'(?P<name>[\w\/\d]+)\{0}(?P<id>\d+)'.format(nameSep[0]))
    else:
        # We have already verified that nameSep is no longer than 2 characters.
        a, b = nameSep 
        p = re.compile(r'(?P<name>[\w\/\d]+)\{0}(?P<id>\d+)\{1}'.format(a, b))
    m = p.match(expr)
    assert m.group('id').isdigit() == True
    return m.group('name'), m.group('id')



def getCompartmentId(compExpr):
    """Get the id of compartment.
    """
    return splitComparmentExpr(compExpr)[1]

def getCompName(compExpr):
    """Get the name of compartment.
    """
    return splitComparmentExpr(compExpr)[0]

def stringToFloat(text):
    text = text.strip()
    if not text:
        return 0.0
    try:
        val = float(text)
        return val
    except Exception:
        raise UserWarning, "Failed to convert {0} to float".format(text)


def dumpMoosePaths(pat, isRoot=True):
    ''' Path is pattern '''
    moose_paths = getMoosePaths(pat, isRoot)
    return "\n\t{0}".format(moose_paths)

def getMoosePaths(pat, isRoot=True):
    ''' Return a list of paths for a given pattern. '''
    if type(pat) != str:
        pat = pat.path
        assert type(pat) == str
    moose_paths = [x.path for x in _moose.wildcardFind(pat)]
    return moose_paths

def dumpMatchingPaths(path, pat='/##'):
    ''' return the name of path which the closely matched with given path 
    pattern pat is optional.
    '''
    a = path.split('/')
    start = a.pop(0)
    p = _moose.wildcardFind(start+'/##')
    common = []
    while len(p) > 0:
        common.append(p)
        start = start+'/'+a.pop(0)
        p = _moose.wildcardFind(start+'/##')
        
    if len(common) > 1:
        matchedPaths = [x.getPath() for x in common[-1]]
    else:
        matchedPaths = []
    return '\n\t'+('\n\t'.join(matchedPaths))


def dumpFieldName(path, whichInfo='valueF'):
    print path.getFieldNames(whichInfo+'info')

"""
TODO: Move it to backend 
def writeGraphviz(pat='/##', filename=None, filterList=[]):
    '''This is  a generic function. It takes the the pattern, search for paths
    and write a graphviz file.
    '''
    def ignore(line):
        for f in filterList:
            if f in line:
                return True
        return False

    pathList = getMoosePaths(pat)
    dot = []
    dot.append("digraph G {")
    dot.append("\tconcentrate=true")
    for p in pathList:
        if ignore(p):
            continue
        else:
            p = p.translate(None, '[]()')
            dot.append('\t'+' -> '.join(filter(None, p.split('/'))))
    dot.append('}')
    dot = '\n'.join(dot)
    if not filename:
        print(dot)
    else:
        with open(filename, 'w') as graphviz:
            print_utils.dump("INFO"
                    , "Writing topology to file {}".format(filename)
                    )
            graphviz.write(dot)
    return 
"""

def setupTable(name, obj, qtyname, tablePath=None, threshold=None):
    '''This is replacement function for moose.utils.setupTable

    It stores qtyname from obj.
    '''
    assert qtyname[0].isupper(), "First character must be uppercase character"
    print_utils.dump("DEBUG"
            , "Setting up table for: {} -> get{}".format(obj.path, qtyname)
            )
    if tablePath is None:
        tablePath = '{}/{}'.format(obj.path, 'data')
        print_utils.dump("WARN"
                , "Using default table path: {}".format(tablePath)
                , frame = inspect.currentframe()
                )
    if not _moose.exists(obj.path):
        raise RuntimeError("Unknown path {}".format(obj.path))

    _moose.Neutral(tablePath)
    table = _moose.Table('{}/{}'.format(tablePath, name))
    if threshold is None:
        _moose.connect(table, "requestOut", obj, "get{}".format(qtyname))
    else:
        raise UserWarning("TODO: Table with threshold is not implemented yet")
    return table

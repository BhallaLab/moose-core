"""behavioural.py: 

Behavioural components.

"""
    
__author__           = "Dilawar Singh"
__copyright__        = "Copyright 2017-, Dilawar Singh"
__version__          = "1.0.0"
__maintainer__       = "Dilawar Singh"
__email__            = "dilawars@ncbs.res.in"
__status__           = "Development"

import _moose
import re
import logging
import ast
import unparser

logger_ = logging.getLogger( 'moose.behav' )

class RewriteName(ast.NodeTransformer):
    # Merge other equations in ODEs.
    def otherASTs(self, astDict):
        self.astDict = astDict

    def visit_Name(self, node):
        if node.id in self.astDict:
            return ast.copy_location(self.astDict[node.id], node)
        return node

def generateAST(eq):
    lhs, rhs = eq.split('=', 1)
    return lhs, ast.parse(rhs)

def _getODEVarName( key ):
    if key[-1] == "'":
        return key[:-1]
    m = re.match( r'd(?P<id>\S+)\/dt', key)
    if m:
        return m.group('id')
    return None

def _flattenAST(a, astDict):
    t = RewriteName()
    t.otherASTs(astDict)
    a = t.visit(a)
    ast.fix_missing_locations(a)
    return a

def _getODESystem(eqs):
    """equations2SS
        X' = AX + B
        Y  = CX + D

    :param eqs:
    :return: Return A, B, C, D
    """
    logger_.info( "Generating state space" )
    astD = dict([generateAST(eq) for eq in eqs])
    odeSys = {}
    for k, v in astD.items():
        # Only modify differential equations.
        var = _getODEVarName(k)
        if var:
            odeSys[r"%s'"%var] = unparser.unparse(
                    _flattenAST(v, astD)
                    ).replace('\n', '')

    # Now build state space description from new dict
    return ["%s=%s"%x for x in odeSys.items()]


class BehavNeuron():
    """BehavNeuron

    A neuron whose behaviour is given by equations.
    """
    
    def __init__(self, path, equations):
        self.nrn = _moose.BehaviouralNeuron(path)
        self.equations = equations
        # StateSpace 
        self.ss = ([],[],[],[])
        self.build()


    def build(self, verbose = True):
        if verbose:
            logger_.setLevel( logging.DEBUG )
        odeEqs = _getODESystem(self.equations)
        logger_.info( "Building behavioural neuron: %s" % odeEqs )
        self.nrn.equations = odeEqs

    


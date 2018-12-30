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

def _replaceConstInLHS(x, **kwargs):
    print( kwargs, x )
    for k, v in kwargs.items():
        if k in x:
            x = x.replace(k, '%s'%v)
    return x

def _getODESystem(eqs, **kwargs):
    """
    Get ODE System after replacing values.
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
    eqs = ["%s=%s"%(l, _replaceConstInLHS(r,**kwargs)) for (l,r) in odeSys.items()]
    return eqs

class BehavNeuron( _moose.BehaviouralNeuron ):
    """BehavNeuron

    A neuron whose behaviour is given by equations.
    """
    def __init__(self, path, equations, **kwargs):
        _moose.BehaviouralNeuron.__init__(self, path)
        self.__eqs__ = equations
        self.build(**kwargs)

    def build(self, **kwargs):
        if kwargs.get('verbose', False):
            logger_.setLevel( logging.DEBUG )
        self.equations = _getODESystem(self.__eqs__, **kwargs)
        logger_.info( "Building behavioural neuron: %s" % self.equations )

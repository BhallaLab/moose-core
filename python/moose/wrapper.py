# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import

# Wrappers around _moose.so classes.

__author__ = "Dilawar Singh"
__copyright__ = "Copyright 2019-, Dilawar Singh"
__maintainer__ = "Dilawar Singh"
__email__ = "dilawars@ncbs.res.in"

import difflib
import moose._moose as _moose
import pprint

import logging
logger = logging.getLogger('moose')

# Provide a pretty printer.
pprint = pprint.pprint

sympyFound_ = False
try:
    import sympy
    sympy.init_printing(use_unicode=True)
    sympyFound_ = True
    # When sympy is found, pprint is sympy pprint.
    pprint = sympy.pprint
except ImportError:
    pass


def _didYouMean(v, options):
    """Did you mean this when you types v
    """
    return ' or '.join(difflib.get_close_matches(v, list(options), n=2))


def _eval(expr):
    try:
        if isinstance(expr, (int, float)):
            return expr
        return eval(str(expr))
    except Exception:
        return expr


def _prettifyExpr(expr):
    global sympyFound_
    if not sympyFound_:
        return expr
    try:
        return sympy.pretty(sympy.simplify(expr, use_unicode=True))
    except Exception:
        return expr


# Generic wrapper around moose.Neutral
class Neutral(_moose.Neutral):
    def __init__(self, path, n=1, g=0, dtype='Neutral', **kwargs):
        super(_moose.Neutral, self).__init__(path, n, g, dtype)
        for k in kwargs:
            try:
                setattr(self, k, kwargs[k])
            except AttributeError:
                logging.warn("Attribute %s is not found. Ignoring..." % k)

    def __new__(cls, pathOrObject, n=1, g=0, dtype='Neutral', **kwargs):
        path = pathOrObject
        if not isinstance(pathOrObject, str):
            path = pathOrObject.path
        if _moose.exists(path):
            #  logger.info("%s already exists. Returning old element."%path)
            return _moose.element(path)
        return super(_moose.Neutral, cls).__new__(cls, pathOrObject, n, g,
                                                  dtype)

    def connect(self, srcField, dest, destField):
        """Wrapper around moose.connect.
        """
        allSrcFields = self.sourceFields
        allDestFields = dest.destFields

        if srcField not in allSrcFields:
            logger.warn("Could not find '{}' in {} sourceFields.".format(
                srcField, self))
            dym = _didYouMean(srcField, allSrcFields)
            if dym:
                logger.warn("\tDid you mean %s?" % dym)
            else:
                logger.warn(': Available options: %s' %
                            ', '.join(allSrcFields))
            raise NameError("Failed to connect")
        if destField not in allDestFields:
            logger.error("Could not find '{0}' in {1} destFields.".format(
                destField, dest))
            dym = _didYouMean(destField, allDestFields)
            if dym:
                logger.warn("\tDid you mean %s?" % dym)
            else:
                logger.warn(': Available options: %s' %
                            ', '.join(allDestFields))
            raise NameError("Failed to connect")

        # Everything looks fine. Connect now.
        _moose.connect(self, srcField, dest, destField)


class Function(_moose.Function):
    """Overides moose._Function

    Provides a convinient way to set expression and connect variables.
    """
    __expr = ""

    def __init__(self, path, n=1, g=0, dtype='Function', **kwargs):
        super(_moose.Function, self).__init__(path, n, g, dtype)
        for k in kwargs:
            try:
                setattr(self, k, kwargs[k])
            except AttributeError:
                logging.warn("Attribute %s is not found. Ignoring..." % k)

    def __getitem__(self, key):
        """Override [] operator. It returns the linked variable by name.
        """
        assert self.numVars > 0
        return self.x[self.xindex[key]]

    def compile(self, expr, constants={}, variables=[], mode=0, **kwargs):
        """Add an expression to a given function.
        """
        __expr = expr
        # Replace constants.
        constants = {
            k: v
            for k, v in sorted(constants.items(), key=lambda x: len(x))
        }
        for i, constName in enumerate(constants):
            # replace constName with 'c%d' % i
            mooseConstName = 'c%d' % i
            expr = expr.replace(constName, mooseConstName)
            self.c[mooseConstName] = _eval(constants[constName])

        self.expr = expr
        self.mode = mode
        if kwargs.get('independent', ''):
            self.independent = kwargs['independent']

        if __expr != expr:
            msg = "Expression has been changed to MOOSE's form."
            msg += "\nFrom,\n"
            msg += _prettifyExpr(__expr)
            msg += "\nto, \n"
            msg += _prettifyExpr(expr)
            logging.warn(msg.replace('\n', '\n ﹅ '))

    # alias
    setExpr = compile

    def sympyExpr(self):
        import sympy
        return sympy.simplify(self.expr)

    def printUsingSympy(self):
        """Print function expression using sympy.
        """
        import sympy
        sympy.pprint(self.sympyExpr())


class StimulusTable(Neutral):
    """StimulusTable
    """
    def __init__(self, path, n=1, g=0, dtype='StimulusTable', **kwargs):
        super(Neutral, self).__init__(path, n, g, dtype)
        for k in kwargs:
            try:
                setattr(self, k, kwargs[k])
            except AttributeError:
                logging.warn("Attribute %s is not found. Ignoring..." % k)

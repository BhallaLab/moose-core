__author__           = "Dilawar Singh"
__copyright__        = "Copyright 2019-, Dilawar Singh"
__maintainer__       = "Dilawar Singh"
__email__            = "dilawars@ncbs.res.in"

import doctest
import moose

print("[INFO ] Using moose from", moose.about())

def test_showfield_func():
    """Show field of a function.
    >>> test_showfield_func()   #doctest: +NORMALIZE_WHITESPACE
    [/dadada[0]]
    allowUnknownVariable    =True
    className               =Function
    derivative              =nan
    doEvalAtReinit          =False
    dt                      =0.1
    expr                    =0
    fieldIndex              =0
    idValue                 =448
    independent             =t
    index                   =0
    mode                    =1
    name                    =dadada
    numData                 =1
    numField                =1
    numVars                 =0
    path                    =/dadada[0]
    rate                    =0.0
    tick                    =12
    useTrigger              =False
    value                   =0.0
    """
    f = moose.Function('/dadada')
    moose.showfield(f)

if __name__ == '__main__':
    import doctest
    doctest.testmod()

#!/usr/bin/env python

"""verification_utils.py:

    IT contains a class which runs tests on moose internal data-structures to
    check if it is good for simulation.

Last modified: Wed May 14, 2014  12:08AM

"""
    
__author__           = "Dilawar Singh"
__copyright__        = "Copyright 2013, NCBS Bangalore"
__credits__          = ["NCBS Bangalore", "Bhalla Lab"]
__license__          = "GPL"
__version__          = "1.0.0"
__maintainer__       = "Dilawar Singh"
__email__            = "dilawars@iitb.ac.in"
__status__           = "Development"


import sys
import sys
import _moose
import unittest
import inspect
import print_utils as debug
import numpy as np

class MooseTestCase( unittest.TestCase ):

    def dump(self, msg, end=''):
        ''' Dump the messages in test functions '''
        caller = inspect.stack()[1][3]
        if type(msg) == list:
            msg = '\n\t|- '.join(msg)
        print('[VERIFY] {:80s}[{}]'.format(msg, caller))
        
    def setUp(self):
        '''Initialize storehouse
        '''
        self.compartments = _moose.wildcardFind('/##[TYPE=Compartment]')
        self.tables = _moose.wildcardFind('/##[TYPE=Table]')
        self.pulse_gens = _moose.wildcardFind('/##[TYPE=PulseGen]')
        self.clocks = _moose.wildcardFind('/##[TYPE=Clock]')
        self.nonZeroClockIds = None

    def test_disconnected_compartments(self):
        '''Test if any comparment is not connected '''
        self.dump("Checking if any compartment is not connected ...")
        for c in self.compartments:
            if (c.neighbors['axial'] or c.neighbors['raxial']):
                continue
            elif c.neighbors['injectMsg']:
                continue
            else:
                msg = '%s is not connected with any other compartment' % c.path
                debug.dump('FAIL'
                        , [ msg
                            , 'Did you forget to use `moose.connect`?'
                            ]
                        )

    def test_isolated_pulse_gen(self):
        ''' Test if any pulse-generator is not injecting current to a
        compartment
        '''
        self.dump('Checking if any pulse-generator is floating')
        for pg in self.pulse_gens:
            if pg.neighbors['output']:
                continue
            else:
                debug.dump(
                        'FAIL'
                        , [ 'Current source {} is floating'.format(pg.path)
                            , 'It is not injecting current to any compartment'
                            , 'Perhaps you forgot to use `moose.connect`?'
                            ]
                        )
    
    def test_unused_tables(self):
        '''Tests if any table is not reading data. Such tables remain empty.
        '''
        self.dump('Checking if any table is not connected')
        for table in self.tables:
            if table.neighbors['requestOut']:
                continue
            else:
                debug.dump(
                        'FAIL'
                        , [ 'Table {} is not reading data.'.format(table.path)
                            , ' Did you forget to use `moose.connect`?'
                            ]
                        )

    def test_clocks(self):
        """Tests if clocks are missing. """
        self.dump("Checking if clocks are available")
        clock = self.clocks[0]
        clockDtList = clock.dts
        if np.count_nonzero(clockDtList) < 1:
            debug.dump("FATAL"
                    , [ "No clock is found with non-zero dt size. "
                        , "Use `moose.setClock` function and confinue."
                        , "Quitting..." 
                        ]
                    )
            sys.exit(0)
        else:
            self.nonZeroClockIds = np.nonzero(self.clocks)

    def test_methods_sensitivity(self):
        """Test if each compartment has process connected to a non-zero clock"""
        self.dump("Checking for insensitive processes")
        [ self.checkSentitivity( m, objs) 
                for m in ['process', 'init']  
                for objs in [self.compartments] 
                ]
        [self.checkSentitivity('process', objs)
                for objs in [self.tables, self.pulse_gens]
                ]


    def checkSentitivity( self, methodName, objectList):
        """Check if a given method is sensitive to any non-zero clock 
        """
        assert type(methodName) == str
        insensitiveObjectList = []
        for obj in objectList:
            if not obj.neighbors[methodName]:
                insensitiveObjectList.append(obj)
            else:
                # Here we must check if method is made sensitive to a
                # zero-clock. Currently there is no way to test it in python.
                pass

        if len(insensitiveObjectList) > 0:
            msgList = [
                    "Method `%s` is insensitive to all clocks. " % methodName
                    , "Total {} out of {} object ({}) fails this test".format(
                        len(insensitiveObjectList)
                        , len(objectList)
                        , type(insensitiveObjectList[0])
                        )
                    ]
            debug.dump("FAIL", msgList)

def verify( *args, **kwargs):
    '''Verify the current moose setup. Emit errors and warnings 
    '''
    connectivitySuite = unittest.TestSuite()
    connectivitySuite.addTest(MooseTestCase('test_disconnected_compartments'))
    connectivitySuite.addTest(MooseTestCase('test_isolated_pulse_gen'))
    connectivitySuite.addTest(MooseTestCase('test_unused_tables'))

    simulationSuite = unittest.TestSuite()
    simulationSuite.addTest(MooseTestCase('test_clocks'))
    simulationSuite.addTest(MooseTestCase('test_methods_sensitivity'))

    # We can replace self with run also and collect the result into a result
    # object.
    connectivitySuite.debug()
    simulationSuite.debug()


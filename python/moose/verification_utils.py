#!/usr/bin/env python

"""verification_utils.py:

    IT contains a class which runs tests on moose internal data-structures to
    check if it is good for simulation.

Last modified: Fri May 09, 2014  02:10PM

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

def verify( *args, **kwargs):
    '''Verify the current moose setup. Emit errors and warnings 
    '''
    connectivitySuite = unittest.TestSuite()
    connectivitySuite.addTest(MooseTestCase('test_disconnected_compartments'))
    connectivitySuite.addTest(MooseTestCase('test_isolated_pulse_gen'))
    connectivitySuite.addTest(MooseTestCase('test_unused_tables'))

    # We can replace self with run also and collect the result into a result
    # object.
    connectivitySuite.debug()


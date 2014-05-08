#!/usr/bin/env python

"""Compartment.py: 

    A compartment in moose.

Last modified: Sat Jan 18, 2014  05:01PM

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
sys.path.append('../../../python')

import moose
import unittest
import math

class MooseCompartment():
    """A simple class for making MooseCompartment in moose"""

    def __init__(self, **kwargs):
        self.mooseCompartment = None
        self.path = None

        # Following values are taken from Upi's chapter on Rallpacks
        self.RM = kwargs.get('RM', 4.0)
        self.RA = kwargs.get('RA', 1.0)
        self.CM = kwargs.get('CM', 0.01)
        self.Em = kwargs.get('Em', -0.060)

    def __repr__( self ):
        msg = '{}: '.format( self.mooseCompartment.path )
        msg += '\n\t|- Ra:{}, Rm:{}, Cm:{}, Em: {}'.format( 
                self.Ra, self.Rm, self.Cm, self.Em
                )
        return msg

    def __str__( self ):
        return self.__repr__( )

    def computeParams( self ):
        '''Compute essentials paramters for compartment. '''

        self.surfaceArea = math.pi * self.diameter * self.length 
        self.crossSection = ( math.pi * self.diameter * self.diameter ) / 4.0
        self.Ra = ( self.RA * self.length ) / self.crossSection
        self.Rm = ( self.RM / self.surfaceArea )
        self.Cm = ( self.CM * self.surfaceArea )
        self.Em = self.Em


    def createCompartment(self, path = None, **kwargs):
        ''' Create a MooseCompartment in moose '''

        self.length = kwargs.get('length', 0.001)
        self.diameter = kwargs.get('diameter', 1.0e-6)
        self.computeParams( )

        try:
            self.mooseCompartment = moose.Compartment(path)
        except Exception as e:
            print("[ERROR] Can't create compartment with path %s " % path)
        self.mooseCompartment.Ra = self.Ra
        self.mooseCompartment.Rm = self.Rm
        self.mooseCompartment.Cm = self.Cm
        self.mooseCompartment.Em = self.Em

class TestCompartment( unittest.TestCase):
    ''' Test class '''

    def setUp( self ):
        self.dut = MooseCompartment()
        self.dut.createCompartment( path = '/dut1' )

    def test_creation( self ):
        m = MooseCompartment( )
        m.createCompartment( path = '/compartment1' )
        self.assertTrue( m.mooseCompartment
                , 'Always create compartments when parent is /.'
                )

        m = MooseCompartment( )
        m.createCompartment( path='/model/compartment1' )
        self.assertFalse ( m.mooseCompartment 
                , 'Should not create compartment when parent does not exists.'
                )
    
    def test_properties( self ):
        m = MooseCompartment()
        m.createCompartment('/comp1')
        self.assertTrue( m.mooseCompartment.Em <= 0.0
                , "Em is initialized to some positive value."
                " Current value is %s " % m.mooseCompartment.Em 
                )
        self.assertTrue( m.mooseCompartment.Rm >= 0.0
                , "Rm should be initialized to non-zero positive float"
                 " Current value is: {}".format( m.mooseCompartment.Rm )
                )

    def test_repr ( self ):
        print( self.dut )

if __name__ == "__main__":
    unittest.main()

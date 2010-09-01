#!/usr/bin/env python

# /*******************************************************************
#  * File:            bulbchan_new.py
#  * Description:      
#  * Author:          Subhasis Ray
#  * E-mail:          ray dot subhasis at gmail dot com
#  * Created:         2008-10-23 10:49:26
#  ********************************************************************/
# /**********************************************************************
# ** This program is part of 'MOOSE', the
# ** Messaging Object Oriented Simulation Environment,
# ** also known as GENESIS 3 base code.
# **           copyright (C) 2003-2008 Upinder S. Bhalla. and NCBS
# ** It is made available under the terms of the
# ** GNU General Public License version 2
# ** See the file COPYING.LIB for the full notice.
#**********************************************************************/

import math

import moose

default_area = 1e-9 # default compartment area

class KMitralUSB(moose.HHChannel):
    """    
    Mitral K current Heavily adapted from : K current activation from
    Thompson, J. Physiol 265, 465 (1977) (Tritonia (LPl 2 and LPl 3
    cells) Inactivation from RW Aldrich, PA Getting, and SH Thompson,
    J. Physiol, 291, 507 (1979)
    """
    e_K = - 0.07

    def __init__(self, *args):
        global default_area
        moose.HHChannel.__init__(self, *args)
        parent = moose.Compartment(self.parent)
        area = default_area
        if parent.diameter > 0.0 and parent.length > 0.0: # comparison to 0.0 is exact
            area = math.pi * parent.diameter * parent.length

        self.Gbar = 1200.0 * area
        self.Ek = KMitralUSB.e_K
        self.Ik = 0.0 
        self.Gk = 0.0 
        self.Xpower = 2
        self.Ypower = 1 
        self.createTable("X", 30, -0.1, 0.05)
        xGate = moose.HHGate(self.path + "/xGate")
        for ii in range(0, 13):
            xGate.A[ii] = 0.0
        xGate.A[14] = 2.87
        xGate.A[15] = 4.68 
        xGate.A[16] = 7.46 
        xGate.A[17] = 10.07 
        xGate.A[18] = 14.27 
        xGate.A[19] = 17.87 
        xGate.A[20] = 22.9  
        xGate.A[21] = 33.6 
        xGate.A[22] = 49.3 
        xGate.A[23] = 65.6  
        xGate.A[24] = 82.0
        xGate.A[25] = 110.0 
        xGate.A[26] = 147.1  
        xGate.A[27] = 147.1
        xGate.A[28] = 147.1 
        xGate.A[29] = 147.1  
        xGate.A[30] = 147.1

        xGate.B[0] = 36.0
        xGate.B[1] = 34.4  
        xGate.B[2] = 32.8
        xGate.B[3] = 31.2
        xGate.B[4] = 29.6  
        xGate.B[5] = 28.0 
        xGate.B[6] = 26.3
        xGate.B[7] = 24.7  
        xGate.B[8] = 23.1 
        xGate.B[9] = 21.5
        xGate.B[10] = 19.9 
        xGate.B[11] = 18.3
        xGate.B[12] = 16.6 
        xGate.B[13] = 15.4 
        xGate.B[14] = 13.5
        xGate.B[15] = 13.2 
        xGate.B[16] = 11.9 
        xGate.B[17] = 11.5
        xGate.B[18] = 10.75
        xGate.B[19] = 9.30 
        xGate.B[20] = 8.30
        xGate.B[21] = 6.00 
        xGate.B[22] = 5.10 
        xGate.B[23] = 4.80
        xGate.B[24] = 3.20 
        xGate.B[25] = 1.60 
        xGate.B[26] = 0.00
        xGate.B[27] = 0.00 
        xGate.B[28] = 0.00 
        xGate.B[29] = 0.00
        xGate.B[30] = 0.00

        
        xGate.A.calcMode = 0 
        xGate.B.calcMode = 0
        self.tweakAlpha("X")
        xGate.tabFill(3000, 0)

        self.createTable("Y", 30, -0.1, 0.05)
        yGate = moose.HHGate(self.path + "/yGate")
        for ii in range(0, 12):
            yGate.A[ii] = 1.0
        
        yGate.A[12] = 1.00 
        yGate.A[13] = 0.97  
        yGate.A[14] = 0.94 
        yGate.A[15] = 0.88 
        yGate.A[16] = 0.75  
        yGate.A[17] = 0.61 
        yGate.A[18] = 0.43 
        yGate.A[19] = 0.305  
        yGate.A[20] = 0.220 
        yGate.A[21] = 0.175 
        yGate.A[22] = 0.155  
        yGate.A[23] = 0.143 
        yGate.A[24] = 0.138 
        yGate.A[25] = 0.137
        yGate.A[26] = 0.136 
        yGate.A[27] = 0.135 
        yGate.A[28] = 0.135 
        yGate.A[29] = 0.135 
        yGate.A[30] = 0.135

        for ii in range(0, 12):
            yGate.B[ii] = 0.0   # -0.1 thru -0.045 => 0.0

        yGate.B[12] = 0.0
        yGate.B[13] = 0.03  
        yGate.B[14] = 0.06 
        yGate.B[15] = 0.12 
        yGate.B[16] = 0.25  
        yGate.B[17] = 0.39 
        yGate.B[18] = 0.57 
        yGate.B[19] = 0.695  
        yGate.B[20] = 0.78 
        yGate.B[21] = 0.825 
        yGate.B[22] = 0.845  
        yGate.B[23] = 0.857 
        yGate.B[24] = 0.862 
        yGate.B[25] = 0.863  
        yGate.B[26] = 0.864 
        yGate.B[27] = 0.865 
        yGate.B[28] = 0.865  
        yGate.B[29] = 0.865 
        yGate.B[30] = 0.865
        # yGate.A.dumpFile("k_ya0.plot")
        # yGate.B.dumpFile("k_yb0.plot")

        # Setting the calc_mode to NO_INTERP for speed 
        yGate.A.calcMode = 0 
        yGate.B.calcMode = 0

        # tweaking the tables for the tabchan calculation
    #     K_mit_usb.tweakAlpha("Y")
        yGate.tweakAlpha()
        # yGate.A.dumpFile("k_ya1.pymoose.plot")
        # yGate.B.dumpFile("k_yb1.pymoose.plot")

        # Filling the tables using B-SPLINE interpolation
        yGate.tabFill(3000, 0)
        # yGate.B.dumpFile("k_yb2.pymoose.plot")

        xGate.A.sy = 5.0 
        xGate.B.sy = 5.0 
        yGate.A.sy = 5.0
        yGate.B.sy = 5.0 
        # xGate.A.dumpFile("k_xa.pymoose.plot")
        # xGate.B.dumpFile("k_xb.pymoose.plot")
        # yGate.A.dumpFile("k_ya.pymoose.plot")
        # yGate.B.dumpFile("k_yb.pymoose.plot")



class NaMitralUSB(moose.HHChannel):
    """Na channel in Mitral Cell"""
    threshold = -0.055     # offset both for resting and for threshold  potential
    e_Na = 0.045 # Na reversal potential

    def __init__(self, *args):
        global default_area
        moose.HHChannel.__init__(self, *args)
        parent = moose.Compartment(self.parent)
        area = default_area
        if parent.length > 0.0 and parent.diameter > 0.0:
            area = math.pi * parent.diameter * parent.length
            
        self.Ek = NaMitralUSB.e_Na # V
        self.Gbar = 1.2e3 * area # S
        self.Ik = 0.0 # A
        self.Gk = 0.0 # S
        self.Xpower = 3
        self.Ypower = 1
        self.Zpower = 0
        self.setupAlpha("X", 320e3 * (0.013 + NaMitralUSB.threshold), 
                        -320e3, 
                        -1.0,
                        -1.0 * (0.013 + NaMitralUSB.threshold), 
                        -0.004, 
                        -280e3 * (0.040 + NaMitralUSB.threshold),
                        280e3, 
                        -1.0, 
                        -1.0 * (0.040 + NaMitralUSB.threshold), 
                        5.0e-3)

        self.setupAlpha("Y", 
                        128.0, 
                        0.0, 
                        0.0, 
                        -1.0 * (0.017 + NaMitralUSB.threshold),
                        0.018, 
                        4.0e3, 
                        0.0, 
                        1.0, 
                        -1.0 * (0.040 + NaMitralUSB.threshold), 
                        -5.0e-3)
        
        xGate = moose.HHGate(self.path + "/xGate")
        # xGate.A.dumpFile("na_xa.pymoose.plot")
        # xGate.B.dumpFile("na_xb.pymoose.plot")
        yGate = moose.HHGate(self.path + "/yGate")
        # yGate.A.dumpFile("na_ya.pymoose.plot")
        # yGate.B.dumpFile("na_yb.pymoose.plot")


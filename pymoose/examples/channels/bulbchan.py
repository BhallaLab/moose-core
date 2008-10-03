#!/usr/bin/env python

#/*******************************************************************
# * File:            bulbchan.py
# * Description:      Some Ca channels for the purkinje cell
# *  L Channel data from :
# *  T. Hirano and S. Hagiwara Pflugers A 413(5) pp463-469, 1989
# * T Channel data from :
# * \
# *   	Kaneda, Wakamori, Ito and Akaike J Neuroph 63(5), pp1046-1051 1990
# * 
# * Originally mplemented by Eric De Schutter - January 1991
# * Converted to NEUROKIT format by Upinder S. Bhalla. Feb 1991
# * Converted to PyMOOSE by Subhasis Ray, 2008
# * 
# * Author:          Subhasis Ray
# * E-mail:          ray dot subhasis at gmail dot com
# * Created:         2008-09-30 15:16:43
# ********************************************************************/
#/**********************************************************************
#** This program is part of 'MOOSE', the
#** Messaging Object Oriented Simulation Environment,
#** also known as GENESIS 3 base code.
#**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
#** It is made available under the terms of the
#** GNU General Public License version 2
#** See the file COPYING.LIB for the full notice.
#**********************************************************************/

import moose

# CONSTANTS
# (I-current)
ECa = 0.07
# (I-current)
ENa = 0.045

SOMA_A = 1e-9 # sq m
# obtain the context element
context = moose.PyMooseBase.getContext()

#/* FILE INFORMATION
#** Rat Na channel, cloned, in oocyte expression system.
#** Data from :
#** Stuhmer, Methfessel, Sakmann, Noda an Numa, Eur Biophys J 1987
#**	14:131-138
#**
#** Expts carred out at 16 deg Celsius.
#** 
#** Implemented in tabchan format by Upinder S. Bhalla March 1991
#** This file depends on functions and constants defined in defaults.g
#*/

#========================================================================
#                        Adjusted LCa channel
#========================================================================
def make_LCa3_mit_usb():
    global context
    global ECa
    path = "LCa3_mit_usb"
    if context.exists():
        LCa3_mit_usb = moose.HHChannel(path)
        return LCa3_mit_usb

    LCa3_mit_usb = HHChannel(path)
    LCa3_mit_usb.Ek = ECa
    LCa3_mit_usb.Gbar = 1200.0*SOMA_A
    LCa3_mit_usb.Ik = 0.0
    LCa3_mit_usb.Gk = 0  
    LCa3_mit_usb.Xpower = 1 
    LCa3_mit_usb.Ypower = 1 
    LCa3_mit_usb.Zpower = 0
    LCa3_mit_usb.setupAlpha("X", 7500.0, 0.0, 1.0, -0.013, -0.007, 1650.0, 0.0, 1.0, -0.014, 0.004)
    LCa3_mit_usb.setupAlpha("Y", 6.8, 0.0, 1.0, 0.030, 0.012, 60.0, 0.0, 1.0, 0.0, -0.011)
    return LCa3_mit_usb



#*********************************************************************
#*                          I-Current (Na)
#********************************************************************/

def make_Na_rat_smsnn(path="Na_rat_smsnn"): 
    """Na current"""
    global context
    global ENa
    if context.exists(path):
        return moose.HHChannel(path)
    

    Na_rat_smsnn = moose.HHChannel(path)
    Na_rat_smsnn.Ek = ENa
    Na_rat_smsnn.Gbar = 1200.0*SOMA_A
    Na_rat_smsnn.Ik = 0.0
    Na_rat_smsnn.Gk = 0.0  
    Na_rat_smsnn.Xpower = 3 
    Na_rat_smsnn.Ypower = 1 
    Na_rat_smsnn.Zpower = 0
    Na_rat_smsnn.createTable("X", 30, -0.1, 0.05)
    xGate = moose.HHGate(Na_rat_smsnn.path()+"/xGate")
    xA = xGate.A
    xB = xGate.B
    xA[0] = 1.0e-4 
    xA[1] = 1.0e-4  
    xA[2] = 1.2e-4 
    xA[3] = 1.45e-4 
    xA[4] = 1.67e-4 
    xA[5] = 2.03e-4 
    xA[6] = 2.47e-4 
    xA[7] = 3.20e-4 
    xA[8] = 3.63e-4  
    xA[9] = 4.94e-4 
    xA[10] = 4.07e-4 
    xA[11] = 4.00e-4
    xA[12] = 3.56e-4
    xA[13] = 3.49e-4
    xA[14] = 3.12e-4
    xA[15] = 2.83e-4
    xA[16] = 2.62e-4
    xA[17] = 2.25e-4
    xA[18] = 2.03e-4
    xA[19] = 1.74e-4
    xA[20] = 1.67e-4
    xA[21] = 1.31e-4
    xA[22] = 1.23e-4
    xA[23] = 1.16e-4
    xA[24] = 1.02e-4
    xA[25] = 0.87e-4
    xA[26] = 0.73e-4
    xA[27] = 0.80e-4
    xA[28] = 0.80e-4
    xA[29] = 0.80e-4
    xA[30] = 0.80e-4
    
    x = -0.1
    dx = 0.15/30.0
    xB = xGate.B
    for i in range(31):
        y = 1.0/(1.0 + exp (-(x + 0.041)/0.0086))
        xB[i] = y
        x = x + dx
    
    Na_rat_smsnn.tweakTau("X")
    xA.calcMode = 0 
    xB.calcMode = 0
    xGate.tabFill(3000, 0)

    Na_rat_smsnn.createTable("Y", 30, -0.1, 0.05)
    yGate = moose.HHGate(Na_rat_smsnn.path() + "/yGate")
    yA = yGate.A
    yA[0] = 0.9e-3 
    yA[1] = 1.0e-3  
    yA[2] = 1.2e-3
    yA[3] = 1.45e-3 
    yA[4] = 1.7e-3  
    yA[5] = 2.05e-3 
    yA[6] = 2.55e-3 
    yA[7] = 3.2e-3 
    yA[8] = 4.0e-3
    yA[9] = 5.0e-3  
    yA[10] = 6.49e-3
    yA[11] = 6.88e-3
    yA[12] = 4.07e-3
    yA[13] = 2.71e-3
    yA[14] = 2.03e-3
    yA[15] = 1.55e-3
    yA[16] = 1.26e-3
    yA[17] = 1.07e-3
    yA[18] = 0.87e-3
    yA[19] = 0.78e-3
    yA[20] = 0.68e-3
    yA[21] = 0.63e-3
    yA[22] = 0.58e-3
    yA[23] = 0.53e-3
    yA[24] = 0.48e-3
    yA[25] = 0.48e-3
    yA[26] = 0.48e-3
    yA[27] = 0.48e-3
    yA[28] = 0.48e-3
    yA[29] = 0.43e-3
    yA[30] = 0.39e-3

    x = -0.1
    dx = 0.15/30.0
    yB = yGate.B
    for i in range(31):
        y = 1.0/(1.0 + exp ((x + 0.064)/0.0102))
        yB[i] = y
        x = x + dx
	
    Na_rat_smsnn.tweakTau("Y")
    yA.calcMode = 0 
    yB.calcMode = 0
    yGate.tabFille(3000, 0)

    return Na_rat_smsnn


def make_Na2_rat_smsnn():
    global context
    if context.exists("Na2_rat_smsnn"):
        return moose.HHChannel("Na2_rat_smsnn")
    EK = -0.07
    Na2_rat_smsnn = make_Na_rat_smsnn("Na2_rat_smsnn")
    xGate = moose.HHGate(Na2_rat_smsnn.path() + "/xGate")
    # The ox field does not exist in MOOSE
    xGate.A.ox = 0.01 
    xGate.B.ox = 0.01 
    yGate = moose.HHGate(Na2_rat_smsnn.path() + "/yGate")
    yGate.A.ox = 0.01  
    yGate.B.ox = 0.01
    return Na2_rat_smsnn

# /**********************************************************************
# **                      Mitral K current
# **  Heavily adapted from :
# **	K current activation from Thompson, J. Physiol 265, 465 (1977)
# **	(Tritonia (LPl	2 and LPl 3 cells)
# ** Inactivation from RW Aldrich, PA Getting, and SH Thompson, 
# ** J. Physiol, 291, 507 (1979)
# **
# **********************************************************************/
def make_K_mit_usb(path="K_mit_usb"): 
    """K-current"""
    print "########## IN make_K_mit_usb"
    if context.exists(path):
        return moose.HHChannel(path)

    EK = -0.07
    K_mit_usb = moose.HHChannel(path)
    K_mit_usb.Ek = EK
    K_mit_usb.Gbar = 1200*SOMA_A
    K_mit_usb.Ik = 0.0 
    K_mit_usb.Gk = 0.0 
    K_mit_usb.Xpower = 2
    K_mit_usb.Ypower = 1 
    K_mit_usb.Zpower = 0

    K_mit_usb.createTable("X", 30, -0.100, 0.050)
    xGate = moose.HHGate(path + "/xGate")
    for i in range(0, 13):
        xGate.A[i] = 0.0        # -0.1 thru -0.045=>0.0

#     // -0.100 Volts
#     // -0.095 Volts
#     // -0.090 Volts
#     // -0.085 Volts
#     // -0.080 Volts
#     // -0.075 Volts
#     // -0.070 Volts
#     // -0.065 Volts
#     // -0.060 Volts
#     // -0.055 Volts
#     // -0.050 Volts
#     // -0.045 Volts
#     // -0.040 Volts
#     // -0.030
#     // -0.020
#     // -0.010
#     // 0.0
#     // 0.010
#     // 0.020
#     // 0.030
#     // 0.040
#     // 0.050

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

#     // -0.100 Volts
#     // -0.095 Volts
#     // -0.090 Volts
#     // -0.085 Volts
#     // -0.080 Volts
#     // -0.075 Volts
#     // -0.070 Volts
#     // -0.065 Volts
#     // -0.060 Volts
#     // -0.055 Volts
#     // -0.050 Volts
#     // -0.045 Volts
#     // -0.040 Volts
#     // -0.030
#     // -0.020
#     // -0.010
#     // 0.00
#     // 0.010
#     // 0.020
#     // 0.030
#     // 0.040
#     // 0.050
    
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
    
    # Setting the calc_mode to NO_INTERP for speed 
    print "Calc modes:", xGate.A.calcMode, xGate.B.calcMode
    xGate.A.calcMode = 0 
    xGate.B.calcMode = 0
    print "####### Calc modes:", xGate.A.calcMode, xGate.B.calcMode
    # tweaking the tables for the tabchan calculation
    K_mit_usb.tweakAlpha("X")

    # Filling the tables using B-SPLINE interpolation 
    xGate.tabFill(3000, 0)
    K_mit_usb.createTable("Y", 30, -0.100, 0.050)
    yGate = moose.HHGate(path + "/yGate")
    for i in range(0, 12):
        yGate.A[i] = 1.0    #-0.1 thru -0.035 => 1.0

#     // -0.040	Volts
#     // 
#     // -0.030	Volts
#     // -0.020
#     // -0.010
#     // 0.00
#     // 0.010
#     // 0.020
#     // 0.030
#     // 0.040
#     // 0.050
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

    for i in range(0, 12):
        yGate.B[i] = 0.0   # -0.1 thru -0.045 => 0.0

#     // -0.040	Volts
#     //
#     // -0.030	Volts
#     // -0.020
#     // -0.010
#     // 0.00
#     // 0.010
#     // 0.020
#     // 0.030
#     // 0.040
#     // 0.050
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

    # Setting the calc_mode to NO_INTERP for speed 
    yGate.A.calcMode = 0 
    yGate.B.calcMode = 0

    # tweaking the tables for the tabchan calculation
    K_mit_usb.tweakAlpha("Y")

    # Filling the tables using B-SPLINE interpolation
    yGate.tabFill(3000, 0)

    xGate.A.sy = 5.0 
    xGate.B.sy = 5.0 
    yGate.A.sy = 5.0
    yGate.B.sy = 5.0 
    K_mit_usb.Ek = EK
    xGate.A.dumpFile("xgate_a.plot")
    xGate.B.dumpFile("xgate_b.plot")
    yGate.A.dumpFile("ygate_a.plot")
    yGate.B.dumpFile("ygate_b.plot")

def make_K2_mit_usb(path="K2_mit_usb"):
	if context.exists(path):
            return moose.HHChannel(path)
        
        EK = -0.07
        K2_mit_usb = make_K_mit_usb(path)
        xGate = moose.HHGate(path + "/xGate")
        yGate = moose.HHGate(path + "/yGate")

        xGate.A.sy = 20.0 
        xGate.B.sy = 20.0 
        yGate.A.sy = 20.0  
        yGate.B.sy = 20.0 
        K2_mit_usb.Ek = EK
        return K2_mit_usb

def make_K_slow_usb(path="K_slow_usb"):
	if context.exists(path):
            return moose.HHChannel(path)

        EK = -0.07

	K_slow_usb = make_K_mit_usb(path)
        xGate = moose.HHGate(path + "/xGate")
        yGate = moose.HHGate(path + "/yGate")
        xGate.A.sy = 1.0
        xGate.B.sy = 1.0
        yGate.A.sy = 1.0
        yGate.B.sy = 1.0
        return K_slow_usb

# //========================================================================
# //			Tabchan Na Mitral cell channel 
# //========================================================================

def make_Na_mit_usb(path="Na_mit_usb"):
    global context
    if context.exists(path):
        return moose.HHChannel(path)


    # offset both for erest and for thresh 
    THRESH = -0.055
    # Sodium reversal potl 
    ENA = 0.045
    Na_mit_usb = moose.HHChannel(path)
    Na_mit_usb.Ek = ENA # V
    Na_mit_usb.Gbar = 1.2e3*SOMA_A # S
    Na_mit_usb.Ik = 0.0 # A
    Na_mit_usb.Gk = 0.0 # S
    Na_mit_usb.Xpower = 3
    Na_mit_usb.Ypower = 1
    Na_mit_usb.Zpower = 0
    Na_mit_usb.setupAlpha("X", 320e3*(0.013 + THRESH), -320e3, -1.0,
                          -1.0*(0.013 + THRESH), -0.004, -280e3*(0.040 + THRESH),
                          280e3, -1.0, -1.0*(0.040 + THRESH), 5.0e-3)

    Na_mit_usb.setupAlpha("Y", 128.0, 0.0, 0.0, -1.0*(0.017 + THRESH),
	     0.018, 4.0e3, 0.0, 1.0, -1.0*(0.040 + THRESH), -5.0e-3)

def make_Na2_mit_usb(path="Na2_mit_usb"):
    global context
    if context.exists(path):
        return moose.HHChannel(path)

    # offset both for erest and for thresh 
    THRESH = -0.060
    # Sodium reversal potl
    ENA = 0.045
    Na2_mit_usb = moose.HHChannel(path)
    Na2_mit_usb.Ek = ENA
    Na2_mit_usb.Gbar = 1.2e3*SOMA_A
    Na2_mit_usb.Ik = 0.0 
    Na2_mit_usb.Gk  = 0.0
    Na2_mit_usb.Xpower = 3 
    Na2_mit_usb.Ypower = 1 
    Na2_mit_usb.Zpower = 0
    Na2_mit_usb.setupAlpha("X", 320e3*(0.013 + THRESH), -320e3, -1.0,
                           -1.0*(0.013 + THRESH), -0.004, -280e3*(0.040 + THRESH),
                           280e3, -1.0, -1.0*(0.040 + THRESH), 5.0e-3)

    Na2_mit_usb.setupAlpha("Y", 128.0, 0.0, 0.0,
                           -1.0*(0.017 + THRESH), 0.018, 4.0e3, 0.0, 1.0,
                           -1.0*(0.040 + THRESH), -5.0e-3)

    return Na2_mit_usb

# TODO: put the rest of the channels


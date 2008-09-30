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

# CONSTANTS
# (I-current)
ECa = 0.07
# (I-current)
ENa = 0.045

SOMA_A = 1e-9 # sq m
# obtain the context element
context = PyMooseBase.getContext()

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

# TODO: put the rest of the channels


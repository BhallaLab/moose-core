#!/usr/bin/python
# python version of bulbchan.g
# Subhasis Ray / 2007-06-11
import math
import moose

def settab2const(gate, table, xdivs, imin, imax, value):
    t = None
    table = table.lstrip().lower()
    if table[0] == 'a':
        t = gate.getA()
    else if table[0] == 'b':
        t = gate.getB()
        
    for i in range(imin, imax+1):
        t.table[i] = value
# end settab2const

ENa = 0.045
SOMA_A = 1e-9

#========================================================================
#                        Adjusted LCa channel
#========================================================================
def make_LCa3_mit_usb():
    if PyMooseBase.exists("LCa3_mit_usb"):
        return
    # (I-current)
    ECa = 0.07
    #    create tabchannel LCa3_mit_usb
    LCa3_mit_usb = moose.HHChannel('LCa3_mit_usb')
    
    #setfield LCa3_mit_usb Ek {ECa} Gbar {1200.0*SOMA_A} Ik 0 Gk 0  \
    #    Xpower 1 Ypower 1 Zpower 0
    LCa3_mit_usb.Ek, LCa3_mit_usb.Gbar, LCa3_mit_usb.Ik, LCa3_mit_usb.Gk, LCa3_mit_usb.Xpower, LCa3_mit_usb.Ypower, LCa3_mit_usb.Zpower = ECa, 1200.0*SOMA_A, 0, 0, 1, 1, 0

    # 	setup_tabchan LCa3_mit_usb X 7500.0 0.0 1.0 -0.013 -0.007 1650.0 \
    # 	     0.0 1.0 -0.014 0.004
    LCa3_mit_usb.setupAlpha('X', 7500.0, 0.0, 1.0, -0.013, -0.007, 1650.0, 0.0, 1.0, -0.014, 0.004)
    # 	setup_tabchan LCa3_mit_usb Y 6.8 0.0 1.0 0.030 0.012 60.0 0.0  \
    # 	    1.0 0.0 -0.011
    LCa3_mit_usb.setupAlpha('Y', 6.8, 0.0, 1.0, 0.030, 0.012, 60.0, 0.0, 1.0, 0.0, -0.011)
    return LCa3_mit_usb
#end make_LCa3_mit_usb


#*********************************************************************
#                          I-Current (Na)
#*********************************************************************

def make_Na_rat_smsnn():
    # (I-current)
    ENa = 0.045
    # 	float x, y, dx
    # 	int i
    # if (({exists Na_rat_smsnn}))
    if moose.PyMooseBase.exists('Na_rat_smsnn'):
        return
        
    # create tabchannel Na_rat_smsnn
    Na_rat_smsnn = moose.HHChannel('Na_rat_smsnn')
    #     setfield Na_rat_smsnn Ek {ENa} Gbar {1200.0*SOMA_A} Ik 0 Gk 0  \
    #         Xpower 3 Ypower 1 Zpower 0
    Na_rat_smsnn.Ek = ENa
    Na_rat_smsnn.Gbar = 1200.0*SOMA_A
    Na_rat_smsnn.Ik = 0
    Na_rat_smsnn.Gk = 0
    Na_rat_smsnn.Xpower = 3
    Na_rat_smsnn.Ypower = 1
    Na_rat_smsnn.Zpower = 0
    
    # call Na_rat_smsnn TABCREATE X 30 -0.1 0.05
    Na_rat_smsnn.createTable('X', 30, -0.01, 0.05)
    
    # -0.100 Volts
    # -0.095 Volts
    # -0.090 Volts
    # -0.085 Volts
    # -0.080 Volts
    # -0.075 Volts
    # -0.070 Volts
    # -0.065 Volts
    # -0.060 Volts
    # -0.055 Volts
    # -0.050 Volts
    # -0.045 Volts
    # -0.040 Volts
    # -0.030
    # -0.020
    # -0.010
    # 0.0
    # 0.010
    # 0.020
    # 0.030
    # 0.040
    # 0.050

    #     setfield Na_rat_smsnn X_A->table[0] 1.0e-4 X_A->table[1] 1.0e-4  \
    #         X_A->table[2] 1.2e-4 X_A->table[3] 1.45e-4 X_A->table[4] 1.67e-4 \
    #          X_A->table[5] 2.03e-4 X_A->table[6] 2.47e-4  \
    #         X_A->table[7] 3.20e-4 X_A->table[8] 3.63e-4  \
    #         X_A->table[9] 4.94e-4 X_A->table[10] 4.07e-4  \
    #         X_A->table[11] 4.00e-4 X_A->table[12] 3.56e-4  \
    #         X_A->table[13] 3.49e-4 X_A->table[14] 3.12e-4  \
    #         X_A->table[15] 2.83e-4 X_A->table[16] 2.62e-4  \
    #         X_A->table[17] 2.25e-4 X_A->table[18] 2.03e-4  \
    #         X_A->table[19] 1.74e-4 X_A->table[20] 1.67e-4  \
    #         X_A->table[21] 1.31e-4 X_A->table[22] 1.23e-4  \
    #         X_A->table[23] 1.16e-4 X_A->table[24] 1.02e-4  \
    #         X_A->table[25] 0.87e-4 X_A->table[26] 0.73e-4  \
    #         X_A->table[27] 0.80e-4 X_A->table[28] 0.80e-4  \
    #         X_A->table[29] 0.80e-4 X_A->table[30] 0.80e-4
    xGate = moose.HHGate('/Na_rat_smsnn/xGate')
    X_A = xGate.getA()
    X_A.table[0] = 1.0e-4
    X_A.table[1] = 1.0e-4  
    X_A.table[2] = 1.2e-4
    X_A.table[3] = 1.45e-4
    X_A.table[4] = 1.67e-4 
    X_A.table[5] = 2.03e-4
    X_A.table[6] = 2.47e-4 
    X_A.table[7] = 3.20e-4
    X_A.table[8] = 3.63e-4 
    X_A.table[9] = 4.94e-4
    X_A.table[10] = 4.07e-4  
    X_A.table[11] = 4.00e-4
    X_A.table[12] = 3.56e-4  
    X_A.table[13] = 3.49e-4
    X_A.table[14] = 3.12e-4  
    X_A.table[15] = 2.83e-4
    X_A.table[16] = 2.62e-4  
    X_A.table[17] = 2.25e-4
    X_A.table[18] = 2.03e-4  
    X_A.table[19] = 1.74e-4
    X_A.table[20] = 1.67e-4  
    X_A.table[21] = 1.31e-4
    X_A.table[22] = 1.23e-4  
    X_A.table[23] = 1.16e-4
    X_A.table[24] = 1.02e-4  
    X_A.table[25] = 0.87e-4
    X_A.table[26] = 0.73e-4  
    X_A.table[27] = 0.80e-4
    X_A.table[28] = 0.80e-4  
    X_A.table[29] = 0.80e-4
    X_A.table[30] = 0.80e-4

    x = -0.1
    dx = 0.15/30.0
    X_B = xGate.getB()
    for i in range(0,31):
        y = 1.0/(1.0 + (exp(-(x + 0.041)/0.0086)))
        #setfield Na_rat_smsnn X_B->table[{i}] {y}
        X_B.table[i] = y
        x = x + dx
	
    #tau_tweak_tabchan Na_rat_smsnn X
    Na_rat_smsnn.tweakTau('X')
    #	setfield Na_rat_smsnn X_A->calc_mode 0 X_B->calc_mode 0
    X_A.calc_mode = 0
    X_B.calc_mode = 0
    #	call Na_rat_smsnn TABFILL X 3000 0
    X_A.tabFill(3000,0)


    #	call Na_rat_smsnn TABCREATE Y 30 -0.1 0.05
    Na_rat_smsnn.createTable('Y', 30, -0.1, 0.05)

    # settab2const(Na_rat_smsnn,Y_A,0,10,6.4e-3)
    #-0.1 thru -0.05=>0.0

    # -0.100 Volts
    # -0.095 Volts
    # -0.090 Volts
    # -0.085 Volts
    # -0.080 Volts
    # -0.075 Volts
    # -0.070 Volts
    # -0.065 Volts
    # -0.060 Volts
    # -0.055 Volts
    # -0.050 Volts
    # -0.045 Volts
    # -0.040 Volts
    # -0.030
    # -0.020
    # -0.010
    # 0.0
    # 0.010
    # 0.020
    # 0.030
    # 0.040
    # 0.050
    #     setfield Na_rat_smsnn Y_A->table[0] 0.9e-3 Y_A->table[1] 1.0e-3  \
    #         Y_A->table[2] 1.2e-3 Y_A->table[3] 1.45e-3 Y_A->table[4] 1.7e-3  \
    #         Y_A->table[5] 2.05e-3 Y_A->table[6] 2.55e-3 Y_A->table[7] 3.2e-3 \
    #          Y_A->table[8] 4.0e-3 Y_A->table[9] 5.0e-3  \
    #         Y_A->table[10] 6.49e-3 Y_A->table[11] 6.88e-3  \
    #         Y_A->table[12] 4.07e-3 Y_A->table[13] 2.71e-3  \
    #         Y_A->table[14] 2.03e-3 Y_A->table[15] 1.55e-3  \
    #         Y_A->table[16] 1.26e-3 Y_A->table[17] 1.07e-3  \
    #         Y_A->table[18] 0.87e-3 Y_A->table[19] 0.78e-3  \
    #         Y_A->table[20] 0.68e-3 Y_A->table[21] 0.63e-3  \
    #         Y_A->table[22] 0.58e-3 Y_A->table[23] 0.53e-3  \
    #         Y_A->table[24] 0.48e-3 Y_A->table[25] 0.48e-3  \
    #         Y_A->table[26] 0.48e-3 Y_A->table[27] 0.48e-3  \
    #         Y_A->table[28] 0.48e-3 Y_A->table[29] 0.43e-3  \
    #         Y_A->table[30] 0.39e-3
    yGate = moose.HHGate(PyMooseBase.pathToId('Na_rat_smsnn/yGate'))
    Y_A = yGate.getA()
    Y_A.table[0] = 0.9e-3
    Y_A.table[1] = 1.0e-3 
    Y_A.table[2] = 1.2e-3
    Y_A.table[3] = 1.45e-3
    Y_A.table[4] = 1.7e-3  
    Y_A.table[5] = 2.05e-3
    Y_A.table[6] = 2.55e-3
    Y_A.table[7] = 3.2e-3 
    Y_A.table[8] = 4.0e-3
    Y_A.table[9] = 5.0e-3  
    Y_A.table[10] = 6.49e-3
    Y_A.table[11] = 6.88e-3  
    Y_A.table[12] = 4.07e-3
    Y_A.table[13] = 2.71e-3  
    Y_A.table[14] = 2.03e-3
    Y_A.table[15] = 1.55e-3  
    Y_A.table[16] = 1.26e-3
    Y_A.table[17] = 1.07e-3  
    Y_A.table[18] = 0.87e-3
    Y_A.table[19] = 0.78e-3  
    Y_A.table[20] = 0.68e-3
    Y_A.table[21] = 0.63e-3  
    Y_A.table[22] = 0.58e-3
    Y_A.table[23] = 0.53e-3  
    Y_A.table[24] = 0.48e-3
    Y_A.table[25] = 0.48e-3  
    Y_A.table[26] = 0.48e-3
    Y_A.table[27] = 0.48e-3  
    Y_A.table[28] = 0.48e-3
    Y_A.table[29] = 0.43e-3  
    Y_A.table[30] = 0.39e-3

    x = -0.1
    dx = 0.15/30.0
    Y_B = yGate.getB()
    for i in range(0,31):
        y = 1.0/(1.0 + (math.exp ((x + 0.064)/0.0102)))
        #		setfield Na_rat_smsnn Y_B->table[{i}] {y}
        Y_B.table[i] = y
        x = x + dx
        
        
    # tau_tweak_tabchan Na_rat_smsnn Y
    Na_rat_smsnn.tweakTau('Y')
    # setfield Na_rat_smsnn Y_A->calc_mode 0 Y_B->calc_mode 0
    Y_A.calc_mode = 0
    Y_B.calc_mode = 0
    # call Na_rat_smsnn TABFILL Y 3000 0
    yGate.tabFill(3000,0)
    return Na_rat_smsnn
#end

#********************************************************************
#*            Transient outward K current
#*******************************************************************/

# CONSTANTS

V_OFFSET = 0.0
VKTAU_OFFSET = 0.0
VKMINF_OFFSET = 0.02
EK = -0.07

def make_KA_bsg_yka():
    if PyMooseBase.exists('KA_bsg_yka'):
        return

    #    create tabchannel KA_bsg_yka
    KA_bsg_yka = moose.HHChannel('KA_bsg_yka')
    #    setfield KA_bsg_yka Ek {EK} Gbar {1200.0*SOMA_A} Ik 0 Gk 0 Xpower 1  \
    #       Ypower 1 Zpower 0
    Ka_bsg_yka.Ek = Ek
    Ka_bsg_yka.Gbar = 1200.0*SOMA_A
    Ka_bsg_yka.Ik = 0
    Ka_bsg_yka.Gk = 0
    Ka_bsg_yka.Xpower = 1
    Ka_bsg_yka.Ypower = 1
    Ka_bsg_yka.Zpower = 0

    # 	setup_tabchan_tau KA_bsg_yka X 1.38e-3 0.0 1.0 -1.0e3 1.0 1.0  \
    # 	    0.0 1.0 {0.042 - V_OFFSET} -0.013
    KA_bsg_yka.setupTau( 'X', 1.38e-3, 0.0, 1.0, -1.0e3, 1.0, 1.0, 0.0, 1.0, (0.042 - V_OFFSET). -0.013)

    # 	setup_tabchan_tau KA_bsg_yka Y 0.150 0.0 1.0 -1.0e3 1.0 1.0 0.0  \
    # 	    1.0 {0.110 - V_OFFSET} 0.018
    KA_bsg_yka.setupTau('Y', 0.150, 0.0, 1.0, -1.0e3, 1.0, 1.0, 0.0, 1.0, (0.110 - V_OFFSET), 0.018)

#end make_KA_bsg_yka



##*******************************************************************
##            Non-inactivating Muscarinic K current
##******************************************************************/
def make_KM_bsg_yka():
    if moose.PyMooseBase.exists('KM_bsg_yka'):
        return
    

    # create tabchannel KM_bsg_yka
    KM_bsg_yka = moose.HHChannel('KM_bsg_yka')
    # setfield KM_bsg_yka Ek {EK} Gbar {1200.0*SOMA_A} Ik 0 Gk 0 Xpower 1  \
    #    Ypower 0 Zpower 0
    KM_bsg_yka.Ek = Ek
    KM_bsg_yka.Gbar = 1200.0*SOMA_A
    KM_bsg_yka.Ik = 0
    KM_bsg_yka.Gk = 0
    KM_bsg_yka.Xpower = 1
    # call KM_bsg_yka TABCREATE X 49 -0.1 0.1
    x = -0.1
    dx = 0.2/49.0
    xGate = moose.HHGate(PyMooseBase.pathToId('KM_bsg_yka/xGate'))
    X_A = xGate.getA()
    X_B = xGate.getB()
    for i in range(50):
        y = 1.0/(3.3*((exp ((x + 0.035 - V_OFFSET)/0.04))) + (math.exp (-(x + 0.035 - V_OFFSET)/0.02)))
        # setfield KM_bsg_yka X_A->table[{i}] {y}
        X_A.table[i] = y
        y = 1.0/(1.0 + (math.exp (-(x + 0.035 - V_OFFSET)/0.01)))
        #setfield KM_bsg_yka X_B->table[{i}] {y}
        X_B.table[i] = y
        x = x + dx

    # tau_tweak_tabchan KM_bsg_yka X
    KM_bsg_yka.tweakTau('X')
    # setfield KM_bsg_yka X_A->calc_mode 0 X_B->calc_mode 0
    X_A.calc_mode = 0
    X_B.calc_mode = 0
    # call KM_bsg_yka TABFILL X 3000 0
    KM_bsg_yka.tabFill('X', 3000, 0)
#end make_KM_bsg_yka()


#**********************************************************************
#*                      Mitral K current
#*  Heavily adapted from :
#*	K current activation from Thompson, J. Physiol 265, 465 (1977)
#*	(Tritonia (LPl	2 and LPl 3 cells)
#* Inactivation from RW Aldrich, PA Getting, and SH Thompson, 
#* J. Physiol, 291, 507 (1979)
#*
#*********************************************************************/
def make_K_mit_usb():# K-current     

    if moose.PyMooseBase.exists('KM_bsg_yka'):
        return

    EK = -0.07

    K_mit_usb = moose.HHChannel('K_mit_usb')
    #     setfield K_mit_usb Ek {EK} Gbar {1200*SOMA_A} Ik 0 Gk 0 Xpower 2  \
    #         Ypower 1 Zpower 0
    K_mit_usb.Ek = EK
    K_mit_usb.Gbar = 1200*SOMA_A
    K_mit_usb.Ik = 0
    K_mit_usb.Gk = 0
    K_mit_usb.Xpower = 2  
    K_mit_usb.Ypower = 1
    K_mit_usb.Zpower = 0

    #call K_mit_usb TABCREATE X 30 -0.100 0.050
    K_mit_usb.createTable('X', 30, -0.100, 0.050)
    
    #settab2const K_mit_usb X_A 0 12 0.0    #-0.1 thru -0.045=>0.0
    # PROB: How to convert the above? check out in the genesis parser
    # SOLN: Turns out to be a userdefined function -
    # see in defaults.g of the bulb model.
    settab2const(moose.HHGate(moose.PyMooseBase.pathToId('K_mit_usb/xGate')), 'A', 0, 12, 0.0)
    # -0.030
    # -0.020
    # -0.010
    # 0.0
    # 0.010
    # 0.020
    # 0.030
    # 0.040
    # 0.050
    #     setfield K_mit_usb X_A->table[13] 0.00 X_A->table[14] 2.87  \
    #         X_A->table[15] 4.68 X_A->table[16] 7.46 X_A->table[17] 10.07  \
    #         X_A->table[18] 14.27 X_A->table[19] 17.87 X_A->table[20] 22.9  \
    #         X_A->table[21] 33.6 X_A->table[22] 49.3 X_A->table[23] 65.6  \
    #         X_A->table[24] 82.0 X_A->table[25] 110.0 X_A->table[26] 147.1  \
    #         X_A->table[27] 147.1 X_A->table[28] 147.1 X_A->table[29] 147.1  \
    #         X_A->table[30] 147.1
    xGate = moose.HHGate(moose.PyMooseBase.pathToId('K_mit_usb/xGate'))
    X_A = xGate.getA()
    X_A.table[13] = 0.00
    X_A.table[14] = 2.87  
    X_A.table[15] = 4.68
    X_A.table[16] = 7.46
    X_A.table[17] = 10.07  
    X_A.table[18] = 14.27
    X_A.table[19] = 17.87
    X_A.table[20] = 22.9  
    X_A.table[21] = 33.6
    X_A.table[22] = 49.3
    X_A.table[23] = 65.6  
    X_A.table[24] = 82.0
    X_A.table[25] = 110.0
    X_A.table[26] = 147.1  
    X_A.table[27] = 147.1
    X_A.table[28] = 147.1
    X_A.table[29] = 147.1  
    X_A.table[30] = 147.1

    # -0.100 Volts
    # -0.095 Volts
    # -0.090 Volts
    # -0.085 Volts
    # -0.080 Volts
    # -0.075 Volts
    # -0.070 Volts
    # -0.065 Volts
    # -0.060 Volts
    # -0.055 Volts
    # -0.050 Volts
    # -0.045 Volts
    # -0.040 Volts
    # -0.030
    # -0.020
    # -0.010
    # 0.00
    # 0.010
    # 0.020
    # 0.030
    # 0.040
    # 0.050
    #     setfield K_mit_usb X_B->table[0] 36.0 X_B->table[1] 34.4  \
    #         X_B->table[2] 32.8 X_B->table[3] 31.2 X_B->table[4] 29.6  \
    #         X_B->table[5] 28.0 X_B->table[6] 26.3 X_B->table[7] 24.7  \
    #         X_B->table[8] 23.1 X_B->table[9] 21.5 X_B->table[10] 19.9  \
    #         X_B->table[11] 18.3 X_B->table[12] 16.6 X_B->table[13] 15.4  \
    #         X_B->table[14] 13.5 X_B->table[15] 13.2 X_B->table[16] 11.9  \
    #         X_B->table[17] 11.5 X_B->table[18] 10.75 X_B->table[19] 9.30  \
    #         X_B->table[20] 8.30 X_B->table[21] 6.00 X_B->table[22] 5.10  \
    #         X_B->table[23] 4.80 X_B->table[24] 3.20 X_B->table[25] 1.60  \
    #         X_B->table[26] 0.00 X_B->table[27] 0.00 X_B->table[28] 0.00  \
    #         X_B->table[29] 0.00 X_B->table[30] 0.00

    X_B = xGate.getB()
    X_B.table[0] = 36.0
    X_B.table[1] = 34.4  
    X_B.table[2] = 32.8
    X_B.table[3] = 31.2
    X_B.table[4] = 29.6  
    X_B.table[5] = 28.0
    X_B.table[6] = 26.3
    X_B.table[7] = 24.7  
    X_B.table[8] = 23.1
    X_B.table[9] = 21.5
    X_B.table[10] = 19.9  
    X_B.table[11] = 18.3
    X_B.table[12] = 16.6
    X_B.table[13] = 15.4  
    X_B.table[14] = 13.5
    X_B.table[15] = 13.2
    X_B.table[16] = 11.9  
    X_B.table[17] = 11.5
    X_B.table[18] = 10.75
    X_B.table[19] = 9.30  
    X_B.table[20] = 8.30
    X_B.table[21] = 6.00
    X_B.table[22] = 5.10  
    X_B.table[23] = 4.80
    X_B.table[24] = 3.20
    X_B.table[25] = 1.60  
    X_B.table[26] = 0.00
    X_B.table[27] = 0.00
    X_B.table[28] = 0.00  
    X_B.table[29] = 0.00
    X_B.table[30] = 0.00

    # 		/* Setting the calc_mode to NO_INTERP for speed */
    # 		setfield K_mit_usb X_A->calc_mode 0 X_B->calc_mode 0
    X_A.calc_mode = 0
    X_B.calc_mode = 0
    # 		/* tweaking the tables for the tabchan calculation */
    # 		tweak_tabchan K_mit_usb X
    # PROB: which function is the above? does not run on genesis parser!
    # SOLN: like setup_tabchan, it defaults to Alpha channel
    K_mit_usb.tweakalpha('X')
    # 		/* Filling the tables using B-SPLINE interpolation */
    # 		call K_mit_usb TABFILL X 3000 0
    xGate.tabFill(3000,0)

    #call K_mit_usb TABCREATE Y 30 -0.100 0.050
    K_mit_usb.createTable('Y', 30, -0.100, 0.050)
    #settab2const K_mit_usb Y_A 0 11 1.0    #-0.1 thru -0.035 => 1.0
    # TODO: Again - how to convert the above?
    settab2const(moose.HHGate(moose.PyMooseBase.pathToId('K_mit_usb/yGate')), 'A', 0, 11, 1.0)
    # -0.040	Volts
    # 
    # -0.030	Volts
    # -0.020
    # -0.010
    # 0.00
    # 0.010
    # 0.020
    # 0.030
    # 0.040
    # 0.050
    #     setfield K_mit_usb Y_A->table[12] 1.00 Y_A->table[13] 0.97  \
    #         Y_A->table[14] 0.94 Y_A->table[15] 0.88 Y_A->table[16] 0.75  \
    #         Y_A->table[17] 0.61 Y_A->table[18] 0.43 Y_A->table[19] 0.305  \
    #         Y_A->table[20] 0.220 Y_A->table[21] 0.175 Y_A->table[22] 0.155  \
    #         Y_A->table[23] 0.143 Y_A->table[24] 0.138 Y_A->table[25] 0.137  \
    #         Y_A->table[26] 0.136 Y_A->table[27] 0.135 Y_A->table[28] 0.135  \
    #         Y_A->table[29] 0.135 Y_A->table[30] 0.135
    yGate = moose.HHGate(moose.PyMooseBase.pathToId('/K_mit_usb/yGate'))
    Y_A = yGate.getA()
    Y_A.table[12] = 1.00
    Y_A.table[13] = 0.97  
    Y_A.table[14] = 0.94
    Y_A.table[15] = 0.88
    Y_A.table[16] = 0.75  
    Y_A.table[17] = 0.61
    Y_A.table[18] = 0.43
    Y_A.table[19] = 0.305  
    Y_A.table[20] = 0.220
    Y_A.table[21] = 0.175
    Y_A.table[22] = 0.155  
    Y_A.table[23] = 0.143
    Y_A.table[24] = 0.138
    Y_A.table[25] = 0.137  
    Y_A.table[26] = 0.136
    Y_A.table[27] = 0.135
    Y_A.table[28] = 0.135  
    Y_A.table[29] = 0.135
    Y_A.table[30] = 0.135

    # settab2const K_mit_usb Y_B 0 11 0.0    #-0.1 thru -0.045 => 0.0
    # TODO: again - how to do this?
    settab2const(moose.HHGate(moose.PyMooseBase.pathToId('K_mit_usb/yGate')),'B', 0, 11, 0.0) 
    # -0.040	Volts
    #
    # -0.030	Volts
    # -0.020
    # -0.010
    # 0.00
    # 0.010
    # 0.020
    # 0.030
    # 0.040
    # 0.050
    #     setfield K_mit_usb Y_B->table[12] 0.0 Y_B->table[13] 0.03  \
    #         Y_B->table[14] 0.06 Y_B->table[15] 0.12 Y_B->table[16] 0.25  \
    #         Y_B->table[17] 0.39 Y_B->table[18] 0.57 Y_B->table[19] 0.695  \
    #         Y_B->table[20] 0.78 Y_B->table[21] 0.825 Y_B->table[22] 0.845  \
    #         Y_B->table[23] 0.857 Y_B->table[24] 0.862 Y_B->table[25] 0.863  \
    #         Y_B->table[26] 0.864 Y_B->table[27] 0.865 Y_B->table[28] 0.865  \
    #         Y_B->table[29] 0.865 Y_B->table[30] 0.865
    Y_B = yGate.getB()
    Y_B.table[12] = 0.0 
    Y_B.table[13] = 0.03      
    Y_B.table[14] = 0.06 
    Y_B.table[15] = 0.12 
    Y_B.table[16] = 0.25          
    Y_B.table[17] = 0.39 
    Y_B.table[18] = 0.57 
    Y_B.table[19] = 0.695          
    Y_B.table[20] = 0.78 
    Y_B.table[21] = 0.825 
    Y_B.table[22] = 0.845  
    Y_B.table[23] = 0.857 
    Y_B.table[24] = 0.862 
    Y_B.table[25] = 0.863  
    Y_B.table[26] = 0.864 
    Y_B.table[27] = 0.865 
    Y_B.table[28] = 0.865  
    Y_B.table[29] = 0.865 
    Y_B.table[30] = 0.865

    #           /* Setting the calc_mode to NO_INTERP for speed */
    # 		setfield K_mit_usb Y_A->calc_mode 0 Y_B->calc_mode 0
    Y_A.calc_mode = 0
    Y_B.calc_mode = 0
    # /* tweaking the tables for the tabchan calculation */
    # 		tweak_tabchan K_mit_usb Y
    # TODO: again, how to convert above? Assuming no specializer means Alpha
    K_mit_usb.tweakAlpha('Y')
    # 		/* Filling the tables using B-SPLINE interpolation */
    # 		call K_mit_usb TABFILL Y 3000 0
    yGate.tabFill(3000, 0)
    # 		setfield K_mit_usb X_A->sy 5.0 X_B->sy 5.0 Y_A->sy 5.0  \
    # 		    Y_B->sy 5.0 Ek {EK}
    X_A.sy = 5.0
    X_B.sy = 5.0
    Y_A.sy = 5.0
    Y_B.sy = 5.0
    K_mit_usb.Ek = Ek

#end make_K_mit_usb


def make_K2_mit_usb():
    if moose.PyMooseBase.exists('K2_mit_usb'):
        return
    EK = -0.07
    ctx = moose.PyMooseBase.getContext()
    if moose.PyMooseBase.exists('K_mit_usb'):
        ctx.move(moose.PyMooseBase.pathToId('K_mit_usb'), 'K2_mit_usb', ctx.getCwe())
        make_K_mit_usb()
    else:
        make_K_mit_usb
        ctx.move(ctx.pathToId('K_mit_usb'), 'K2_mit_usb', ctx.getCwe())


        # 	setfield K2_mit_usb X_A->sy 20.0 X_B->sy 20.0 Y_A->sy 20.0  \
        # 	    Y_B->sy 20.0 Ek {EK}
        xGate = moose.HHGate(ctx.pathToId('K2_mit_usb/xGate'))
        X_A = xGate.getA()
        X_B = xGate.getB()
        yGate = moose.HHGate(ctx.pathToId('K2_mit_usb/yGate'))
        Y_A = yGate.getA()
        Y_B = yGate.getB()
        X_A.sy = 20.0
        X_B.sy = 20.0
        Y_A.sy = 20.0  
        Y_B.sy = 20.0
        K2_mit_usb.Ek = EK

#end make_K2_mit_usb

def make_K_slow_usb():
    ctx = moose.PyMooseBase.getContext()
    if ctx.exists('K_slow_usb'):
        return

    EK = -0.07

    if ctx.exists('K_mit_usb'):
        ctx.move(ctx.pathToId('K_mit_usb', 'K_slow_usb', ctx.getCwe()))
        make_K_mit_usb()
    else:
        make_K_mit_usb()
        ctx.move(ctx.pathToId('K_mit_usb', 'K_slow_usb', ctx.getCwe()))

        #         setfield K_slow_usb X_A->sy 1.0 X_B->sy 1.0 Y_A->sy 1.0  \
        # 	    Y_B->sy 1.0
        xGate = moose.HHGate(ctx.pathToId('K_slow_usb/xGate'))
        yGate = moose.HHGate(ctx.pathToId('K_slow_usb/yGate'))
        X_A = xGate.getA()
        X_B = xGate.getB()
        Y_A = yGate.getA()
        Y_B = yGate.getB()
#end make_K_slow_usb


#========================================================================
#			Tabchan Na Mitral cell channel 
#========================================================================

def make_Na_mit_usb():
    ctx = moose.PyMooseBase.getContext()
    if ctx.exists('Na_mit_usb'):
        return


    #/* offset both for erest and for thresh */
    THRESH = -0.055
    #/* Sodium reversal potl */
    ENA = 0.045

    Na_mit_usb = moose.HHChannel('Na_mit_usb')
    #	V
    #	S
    #	A
    #	S
    # 		setfield ^ Ek {ENA} Gbar {1.2e3*SOMA_A} Ik 0 Gk 0  \
    # 		    Xpower 3 Ypower 1 Zpower 0
    # TODO: what dows the caret mean?
    
    # 	setup_tabchan Na_mit_usb X {320e3*(0.013 + THRESH)} -320e3 -1.0  \
    # 	    {-1.0*(0.013 + THRESH)} -0.004 {-280e3*(0.040 + THRESH)}  \
    # 	    280e3 -1.0 {-1.0*(0.040 + THRESH)} 5.0e-3
    Na_mit_usb.setupAlpha('X', 320e3*(0.013+THRESH), -320e3, -1.0, -1.0*(0.013+THRESH), -0.004, -280e3*(0.040+THRESH), 280e3, -1.0, -1.0*(0.040+THRESH), 5.0e-3)
    # 	setup_tabchan Na_mit_usb Y 128.0 0.0 0.0 {-1.0*(0.017 + THRESH)} \
    # 	     0.018 4.0e3 0.0 1.0 {-1.0*(0.040 + THRESH)} -5.0e-3
    Na_mit_usb.setupAlpha('Y',128.0, 0.0, 0.0, -1.0*(0.017 + THRESH), 0.018, 4.0e3, 0.0, 1.0, -1.0*(0.040 + THRESH), -5.0e-3)
# end make_Na_mit_usb

#========================================================================

def make_Na2_mit_usb():
    ctx = moose.PyMooseBase.getContext()
    if ctx.exists('Na2_mit_usb'):
        return
    
    # /* offset both for erest and for thresh */
    THRESH = -0.060
    # /* Sodium reversal potl */
    ENA = 0.045

    #create tabchannel Na2_mit_usb
    Na2_mit_usb = moose.HHChannel('Na2_mit_usb')
    #	V
    #	S
    #	A
    #	S
    # 		setfield ^ Ek {ENA} Gbar {1.2e3*SOMA_A} Ik 0 Gk 0  \
    # 		    Xpower 3 Ypower 1 Zpower 0
    # TODO: again, what does the caret mean?
    # 	setup_tabchan Na2_mit_usb X {320e3*(0.013 + THRESH)} -320e3 -1.0 \
    # 	     {-1.0*(0.013 + THRESH)} -0.004 {-280e3*(0.040 + THRESH)}  \
    # 	    280e3 -1.0 {-1.0*(0.040 + THRESH)} 5.0e-3
    Na2_mit_usb.setupAlpha('X',320e3*(0.013 + THRESH), -320e3, -1.0, -1.0*(0.013 + THRESH), -0.004, -280e3*(0.040 + THRESH), 280e3, -1.0, -1.0*(0.040 + THRESH), 5.0e-3)
    # 	setup_tabchan Na2_mit_usb Y 128.0 0.0 0.0  \
    # 	    {-1.0*(0.017 + THRESH)} 0.018 4.0e3 0.0 1.0  \
    # 	    {-1.0*(0.040 + THRESH)} -5.0e-3
    Na2_mit_usb.setupAlpha('Y', 128.0, 0.0, 0.0, -1.0*(0.017 + THRESH), 0.018, 4.0e3, 0.0, 1.0, -1.0*(0.040 + THRESH), -5.0e-3)

#end make_Na2_mit_usb

#========================================================================
# CONSTANTS
EGlu = 0.045
EGABA_1 = -0.080
EGABA_2 = -0.080
SOMA_A = 1e-9
GGlu = SOMA_A*50
GGABA_1 = SOMA_A*50
GGABA_2 = SOMA_A*50

#===================================================================
#                     SYNAPTIC CHANNELS   (Values guessed at)
#===================================================================


def make_glu_mit_usb():
    ctx = moose.PyMooseBase.getContext()
    if ctx.exists('glu_mit_usb'):
        return

    # for receptor input only
    # create channelC2 glu_mit_usb
    # TODO: What is this? genesis_parser cannot resolve this class either
    # sec
    # sec
    # Siemens
    # 	setfield glu_mit_usb Ek {EGlu} tau1 {4.0e-3} tau2 {4.0e-3}  \
    #     	    gmax {GGlu}
# end make_glu_mit_usb

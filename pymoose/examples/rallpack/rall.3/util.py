#!/bin/python

import moose
import math

context = moose.PyMooseBase.getContext()
def calc_esl(form, v, A, B, V0):
	if ( form == 1 ):
		return A * math.exp((v-V0)/B)
        if ( form == 2 ):
		return A / ( 1.0 + math.exp(( v - V0 ) / B ))
        if ( form == 3):
		if  ( math.fabs( v - V0 ) < 1e-6 ):
			v = v + 1e-6
		return A*(v-V0)/(math.exp((v-V0)/B) - 1.0 )

def calc_Na_m_alpha(v):
        form = 3
        A = - 1.0e5
        B = -0.010
        V0 = -0.040
        return calc_esl( form, v, A, B, V0)

def calc_Na_m_beta(v):
        form = 1
        A = 4.0e3
        B = -0.018
        V0 = - 0.065
        return calc_esl(form, v, A, B, V0)

def calc_Na_h_alpha(v):
        form = 1
        A = 70.0
        B = -0.020
        V0 = - 0.035
        return calc_esl(form, v, A, B, V0)

def calc_Na_h_beta(v):
        form = 2
        A = 1.0e3
        B = -0.010
        V0 = -0.035
        return calc_esl(form, v, A, B, V0)

def calc_K_n_alpha(v):
        form = 3
        A = -1.0e4
        B = -0.010
        V0 = -0.055
        return calc_esl(form, v, A, B, V0)

def calc_K_n_beta(v):
        form = 1
        B = -0.080
        V0 = -0.065
        return calc_esl(form, v, A, B, V0)

def make_compartment(path, RA, RM, CM, EM, inject, diameter, length):
        PI_D_L = math.pi*diameter*length
        Ra = 4.0 + length * RA / PI_D_L
        Rm = RM / PI_D_L
        Cm = CM * PI_D_L
        comp = moose.Compartment(path)
        comp.Ra = Ra
        comp.Rm = Rm
        comp.Cm = Cm
        comp.Em = EM
        comp.inject = inject
        comp.diameter = diameter
        comp.length = length
        comp.initVm = EM

        ENa = 0.050
        GNa = 1200
        EK = - 0.077
        GK = 360
        Gbar_Na = GNa * PI_D_L
        Gbar_K = GK * PI_D_L
        NaChan = moose.HHChannel(path+'/Na')
        NaChan.Ek = ENa
        NaChan.Gbar = Gbar_Na
        NaChan.Xpower = 3
        NaChan.Ypower = 1
        NaChan.X = 0.05293250
        NaChan.Y = 0.59612067
        KChan = moose.HHChannel(path+'/K')
        KChan.Ek = EK
        KChan.Gbar = Gbar_K
        KChan.Xpower = 4
        KChan.X = 0.31767695
        parent = NaChan.parent
        context.connect(parent, 'channel', NaChan, 'channel')
        context.connect(parent, 'channel', KChan, 'channel')

        VMIN = - 0.100
        VMAX = 0.05
        NDIVS = 150
	
	Na_xGate_A = moose.Table(NaChan.path+'/xGate/A')
	Na_xGate_A.xmin = VMIN
	Na_xGate_A.xmax = VMAX
	Na_xGate_A.xdivs = NDIVS
	Na_xGate_B = moose.Table(NaChan.path+'/xGate/B')
	Na_xGate_B.xmin = VMIN
	Na_xGate_B.xmax = VMAX
	Na_xGate_B.xdivs = NDIVS
	Na_yGate_A = moose.Table(NaChan.path+'/yGate/A')
	Na_yGate_A.xmin = VMIN
	Na_yGate_A.xmax = VMAX
	Na_yGate_A.xdivs = NDIVS
	Na_yGate_B = moose.Table(NaChan.path+'/yGate/B')
	Na_yGate_B.xmin = VMIN
	Na_yGate_B.xmax = VMAX
	Na_yGate_B.xdivs = NDIVS
	K_xGate_A = moose.Table(KChan.path+'/xGate/A')
	K_xGate_A.xmin = VMIN
	K_xGate_A.xmax = VMAX
	K_xGate_A.xdivs = NDIVS
	K_xGate_B = moose.Table(KChan.path+'/xGate/B')
	K_xGate_B.xmin = VMIN
	K_xGate_B.xmax = VMAX
	K_xGate_B.xdivs = NDIVS
	
	v = VMIN
	dv = (VMAX-VMIN)/NDIVS

	for i in range(NDIVS+1):
		Na_xGate_A[i] = calc_Na_m_alpha(v)
		Na_xGate_B[i] = calc_Na_m_alpha(v) + calc_Na_m_beta(v)
		Na_yGate_A[i] = calc_Na_h_alpha(v)
		Na_yGate_B[i] =  calc_Na_h_alpha(v) + calc_Na_h_beta(v)
		K_xGate_A[i] = calc_K_n_alpha(v)
		K_xGate_B[i] = calc_K_n_alpha(v)+calc_K_n_beta(v)
		v = v + dv

def link_compartment(path1, path2):
	context.connect(Id(path1),'raxial', Id(path2), 'axial')

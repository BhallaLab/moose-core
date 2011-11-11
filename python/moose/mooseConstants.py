# -*- coding: utf-8 -*-
## Table step_mode
TAB_IO=0 # table acts as lookup - default mode
TAB_ONCE=2 # table outputs value until it reaches the end and then stays at the last value
TAB_BUF=3 # table acts as a buffer: succesive entries at each time step
TAB_SPIKE=4 # table acts as a buffer for spike times. Threshold stored in the pymoose 'stepSize' field.

## Table fill modes
BSplineFill = 0 # B-spline fill (default)
CSplineFill = 1 # C_Spline fill (not yet implemented)
LinearFill = 2 # Linear fill

## The hsolve and ee methods use clock 1.
## hsolve further uses clock 2 for mg_block, nmdachan and others.
## keep clock 3 for plots.
PLOTCLOCK = 3

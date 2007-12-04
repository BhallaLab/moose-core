#!/usr/bin/env python
######################################################################
# File:         wxSquid.py
#
# Description: This is a script showing a small user interface using
#               wxPython to run the squid demo of MOOSE and plot the
#               membrane potential.
#
# Author:       Subhasis Ray
# Date:         2007-12-04 17:16:22
# Email:        ray dot subhasis at gmail dot com
# Copyleft:     Subhasis Ray, NCBS, Bangalore
# License:      GPL-3 [http://www.gnu.org/licenses/gpl-3.0.html]
######################################################################
import sys

# The PYTHONPATH should contain the location of moose.py and _moose.so
# files.  Putting ".." with the assumption that moose.py and _moose.so
# has been generated in ${MOOSE_SOURCE_DIRECTORY}/pymoose/ (as default
# pymoose build does) and this file is located in
# ${MOOSE_SOURCE_DIRECTORY}/pymoose/examples
sys.path.append('..')
try:
    import moose
except ImportError:
    print "ERROR: Could not import moose. Please add the directory containing moose.py in your PYTHONPATH"
    import sys
    sys.exit(1)

import math
VMIN = -0.1
VMAX = 0.05
NDIVS = 150
v = VMIN
dv = ( VMAX - VMIN ) / NDIVS
SIMDT = 1e-5
PLOTDT = 1e-4
RUNTIME = 0.5
EREST = -0.07
VLEAK = EREST + 0.010613
VK = EREST -0.012
VNa = EREST + 0.115
RM = 424.4e3
RA = 7639.44e3
GLEAK = 0.3e-3
GK = 36e-3
GNa = 120e-3
CM = 0.007854e-6
INJECT = 0.1e-6

GK = 0.282743e-3
GNa = 0.94248e-3

def calc_Na_m_A( v ):
    if math.fabs(EREST+0.025-v) < 1e-6:
        v = v + 1e-6
    return 0.1e6 * (EREST + 0.025 -v)/(math.exp((EREST + 0.025 - v)/0.01) - 1.0)

def calc_Na_m_B(v):
    return 4.0e3 * math.exp((EREST - v)/0.018)

def calc_Na_h_A( v ):
    return  70.0 * math.exp(( EREST - v )/0.020 )

def calc_Na_h_B( v ):
    return ( 1.0e3 / (math.exp ( ( 0.030 + EREST - v )/ 0.01 ) + 1.0 ))

def calc_K_n_A( v ):
    if math.fabs( 0.01 + EREST - v )  < 1.0e-6 :
        v = v + 1.0e-6
    return ( 1.0e4 * ( 0.01 + EREST - v ) )/(math.exp(( 0.01 + EREST - v )/0.01) - 1.0 )

def calc_K_n_B( v ):
	return 0.125e3 * math.exp((EREST - v ) / 0.08 )

# Make the squid object and Vm object global

context = moose.PyMooseBase.getContext()
Vm = None

def setupModel():
    """Set up the MOOSE model for squid demo"""
    global Vm
    global context

    if context.exists('squid'):
        Vm = moose.Table('Vm')
        return moose.Compartment('squid')
    squid = moose.Compartment('squid')
    squid.Rm = RM
    squid.Ra = RA
    squid.Cm = CM
    squid.Em = VLEAK

    Na =  moose.HHChannel('Na', squid)
    Na.Ek = VNa
    Na.Gbar = GNa
    Na.Xpower = 3
    Na.Ypower = 1
    
    K = moose.HHChannel('K', squid)
    K.Ek = VK
    K.Gbar = GK
    K.Xpower = 4

    squid.connect('channel', Na, 'channel')
    squid.connect('channel', K, 'channel')
    
    Vm = moose.Table('Vm')
    Vm.stepmode = 3
    Vm.connect('inputRequest', squid, 'Vm')
    
    Na_xA = moose.InterpolationTable('/squid/Na/xGate/A')
    Na_xB = moose.InterpolationTable('/squid/Na/xGate/B')
    Na_yA = moose.InterpolationTable('/squid/Na/yGate/A')
    Na_yB = moose.InterpolationTable('/squid/Na/yGate/B')
    K_xA = moose.InterpolationTable('/squid/K/xGate/A')
    K_xB = moose.InterpolationTable('/squid/K/xGate/B')

   
    Na_xA.xmin = VMIN
    Na_xA.xmax = VMAX
    Na_xA.xdivs = NDIVS

    Na_xB.xmin = VMIN
    Na_xB.xmax = VMAX
    Na_xB.xdivs = NDIVS

    Na_yA.xmin = VMIN
    Na_yA.xmax = VMAX
    Na_yA.xdivs = NDIVS

    Na_yB.xmin = VMIN
    Na_yB.xmax = VMAX
    Na_yB.xdivs = NDIVS

    K_xA.xmin = VMIN
    K_xA.xmax = VMAX
    K_xA.xdivs = NDIVS

    K_xB.xmin = VMIN
    K_xB.xmax = VMAX
    K_xB.xdivs = NDIVS
    
    v = VMIN

    for i in range(NDIVS+1):
	Na_xA[i] = calc_Na_m_A ( v )
	Na_xB[i]  =  calc_Na_m_A (v)   +  calc_Na_m_B ( v   )
	Na_yA[i] = calc_Na_h_A  (v )
	Na_yB[i] =  calc_Na_h_A  (v)   +   calc_Na_h_B  (v   )
	K_xA[i] = calc_K_n_A  (v )
	K_xB[i] =  calc_K_n_A ( v)   +  calc_K_n_B ( v )
	v = v + dv

    context = moose.PyMooseBase.getContext()
    context.setClock(0, SIMDT, 0)
    context.setClock(1, PLOTDT, 0)
    context.useClock(moose.PyMooseBase.pathToId('/sched/cj/t0'), '/Vm,/squid,/squid/#')

    squid.initVm = EREST
    return squid


def runDemo():
    """Run the simulation steps, setup the model if required"""

    global Vm
    squid = setupModel()
    context.reset()
    squid.inject = 0
    context.step(0.005)
    squid.inject = INJECT
    context.step(0.040)
    squid.inject = 0
    context.step(0.005)
    Vm.dumpFile("squid.plot")


##############################################################
# THE FOLLOWING PART IS FOR TESTING OUT wxPython
##############################################################

import wx
from wx.lib.floatcanvas import NavCanvas, FloatCanvas
import sys

try:
    import wx.lib.plot as wxplot
except ImportError:
    print "Could not find wx.lib.plot"
    sys.exit(1)

ID_EXIT=1000
ID_RUN= 1001
ID_CLEAR=1002
ID_SAVE=1003

class SquidFrame(wx.Frame):
    """A very simple frame object for Squid Demo, extends wx.Frame class. It has a File menu with the menu items Exit, Run, Clear and Save plot. It contains a PlotCanvas where the membrane potential Vm is plotted."""
    global Vm
    def __init__(self, parent, ID, title):
        wx.Frame.__init__(self, parent, ID, title)
        self.CreateStatusBar()
        self.SetStatusText("This is a status bar ")
        self.menu = wx.Menu()
        self.menu.Append(ID_EXIT, "E&xit", "Quit")
        self.menu.Append(ID_RUN, "R&un", "Run simulation")
        self.menu.Append(ID_CLEAR, "C&lear", "Clear the canvas")
        self.menu.Append(ID_SAVE, "S&ave", "Save plot")
        self.menuBar = wx.MenuBar()
        self.menuBar.Append(self.menu, "&File")
        self.SetMenuBar(self.menuBar)
        
        # Add the Canvas
        self.canvas = wxplot.PlotCanvas(parent=self, id=wx.ID_ANY, name="PlotCanvas")
        wx.EVT_MENU(self, ID_EXIT, self.Exit)
        wx.EVT_MENU(self, ID_RUN, self.Run)
        wx.EVT_MENU(self, ID_CLEAR, self.Clear)
        wx.EVT_MENU(self, ID_SAVE, self.Save)

    def Exit(self, event):
        self.Close(True)
        
    def Run(self, event):
        """Run the squid demo, setting up the model if required, and plot the Vm"""
        runDemo()
        self.Replot()
        
    def Clear(self, event):
        """Clear the canvas"""
        self.canvas.Clear()


    def Save(self, event):
        """Save the plot in a file"""
        self.canvas.SaveFile()

    def Replot(self):
        """Replot Vm"""
        global Vm
        data = []
        time = 0.0
        # Create data = [(t0, v0), (t1,v1), (t2,v2), ... ]
        for value in Vm:
            data.append((time,value))
            time = time + SIMDT
        self.canvas.Clear()
        # Create a new PolyLine object from the data points
        plot = wxplot.PolyLine(data)
        # Create a PlotGraphics object with title "Membrane Potential", xlabel "s" and ylabel "V"
        graphics = wxplot.PlotGraphics([plot],"Membrane Potential", "s", "V")
        # Actually draw stuff on canvas
        self.canvas.Draw(graphics)
        # Enable zooming on the plot
        self.canvas.SetEnableZoom(True)
        print "Replotting data"


class SquidGui(wx.App):
    """A simple subclass of wx.App"""
    def OnInit(self):
        frame = SquidFrame(None, -1, "Squid Demo")
        frame.Show(True)
        self.SetTopWindow(frame)
        return True

def main():
    app = SquidGui()
    app.MainLoop()

if __name__ == '__main__':
    main()

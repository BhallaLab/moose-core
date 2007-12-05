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

##############################################################
# THE FOLLOWING PART IS FOR TESTING OUT wxPython
##############################################################

import wx
from wx.lib.floatcanvas import NavCanvas, FloatCanvas
import sys
sys.path.append('.')
try:
    import wx.lib.plot as wxplot
except ImportError:
    print "Could not find wx.lib.plot"
    sys.exit(1)

import squid
SIMDT = 1e-5
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
        self.Vm = None
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
        self.Vm = squid.runDemo()
        self.Replot()
        
    def Clear(self, event):
        """Clear the canvas"""
        self.canvas.Clear()


    def Save(self, event):
        """Save the plot in a file"""
        self.canvas.SaveFile()

    def Replot(self):
        """Replot Vm"""
        data = []
        time = 0.0
        # Create data = [(t0, v0), (t1,v1), (t2,v2), ... ]
        for value in self.Vm:
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

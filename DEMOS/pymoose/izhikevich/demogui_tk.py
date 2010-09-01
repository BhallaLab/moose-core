# demogui_tk.py --- 
# 
# Filename: demogui_tk.py
# Description: 
# Author: Subhasis Ray
# Maintainer: 
# Created: Thu Jun 17 08:54:16 2010 (+0530)
# Version: 
# Last-Updated: Tue Jun 22 14:09:52 2010 (+0530)
#           By: Subhasis Ray
#     Update #: 770
# URL: 
# Keywords: 
# Compatibility: 
# 
# 

# Commentary: 
# 
# This is a pure Tkinter based GUI for demonstrating Izhikevich
# models.  This should run on any python distribution with Tkinter
# installed (default on Windows and Mac).
# The original Izhikevich module relies on numpy.
# 

# Change log:
# 
# 
# 
# 
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation; either version 3, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program; see the file COPYING.  If not, write to
# the Free Software Foundation, Inc., 51 Franklin Street, Fifth
# Floor, Boston, MA 02110-1301, USA.
# 
# 

# Code:

import numpy
from Tkinter import *
import tkMessageBox
from Izhikevich import IzhikevichDemo

class CoordinateXform:
    """
    Transform coordinates from world to screen and vice versa.

    xmin - the coordinate (cartesian/world) of the leftmost point of x axis.

    ymin - the coordinate (cartesian/world) of the leftmost point of y axis.

    xmax - the coordinate (cartesian/world) of the rightmost point of x axis.
                                                   
    ymax - the coordinate (cartesian/world) of the rightmost point of y axis.

    parent - the widget for which this transform works.

    """
    # There is a problem here - the current axes are
    # overspecified. Only position of (0,0) and (1,1) in screen
    # coordinates is sufficient.
    def __init__(self, xmin, ymin, xmax, ymax, parent):
        """
        (xmin, ymin) - coordinate of bottom left.
        (xmax, ymax) - coordinate of top right."""
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.parent = parent

    @property
    def xscale(self):
        return self.width * 1.0 / (self.xmax - self.xmin)

    @property
    def yscale(self):
        return self.height * 1.0 / (self.ymax - self.ymin)

    @property
    def width(self):
        return self.parent.winfo_width()

    @property
    def height(self):
        return self.parent.winfo_height()

    def screen(self, x, y):
        """Convert cartesian coordinates to screen coordinates"""
    
        xlist = (x - self.xmin) * self.xscale
        ylist = self.height - (y - self.ymin) * self.yscale
        if (isinstance(xlist, int) or isinstance(xlist, float)) and (isinstance(ylist, int) or isinstance(ylist, float)):
            ret = (int(xlist), int(ylist))
            return ret
        elif isinstance(xlist, numpy.ndarray) and isinstance(ylist, numpy.ndarray):
            return xlist.astype(numpy.int32), ylist.astype(numpy.int32)
        else:
            print 'Could not recognize data type of input coordinate: returning (None, None)'
            return (None, None)
        
    def cartesian(self, x, y):
        """Convert screen coordinates to cartesian."""
        return x / self.xscale + self.xmin,  (self.height - y) / self.yscale + self.ymin

        
class Tkplot(Canvas):
    def __init__(self, *args):
        Canvas.__init__(self, *args)
        self.axes = CoordinateXform(0.0, 0.0, 100.0, 100.0, self)
        self.ticklabel = True
        self.xlabel = ''
        self.ylabel = ''
        self.data = []
        self.hold = False
        
    def plot(self, xlist, ylist, color='blue'):
        if len(xlist) != len(ylist):
            raise Exception, 'x and y lists are not same size'
        self.data.append((xlist, ylist, color))    

    def show(self):
        self.delete(ALL)
        self.draw_axes()
        for data in self.data:
            xlist = numpy.array(data[0])
            ylist = numpy.array(data[1])
            color = data[2]
            (x_screen, y_screen) = self.axes.screen(xlist, ylist)
            positions = numpy.dstack((x_screen, y_screen)).flatten().tolist()
            self.create_line(positions, fill=color)
        self.update()

    def clear(self):
        self.data = []
        self._do_update(None)

    def set_axes(self, xmin, ymin, xmax, ymax, xinterval, yinterval, color='black', ticklabel=True):
        self.ticklabel = ticklabel
        self.axes.xmin = xmin
        self.axes.ymin = ymin
        self.axes.xmax = xmax
        self.axes.ymax = ymax
        self.xinterval = xinterval
        self.yinterval = yinterval
        self.axiscolor = color

    def draw_axes(self):
        (x0, y0) = self.axes.screen(0, 0)
        (xstart, ystart) = self.axes.screen(self.axes.xmin, self.axes.ymin)
        (xend, yend) = self.axes.screen(self.axes.xmax, self.axes.ymax)
        self.xaxis = self.create_line(xstart, y0, xend, y0, fill=self.axiscolor)
        self.yaxis = self.create_line(x0, ystart, x0, yend,  fill=self.axiscolor)
        self.itemconfig(self.xaxis, tags=('x'))
        self.itemconfig(self.yaxis, tags=('y'))
        x = self.axes.xmin
        while x < self.axes.xmax:
            position = self.axes.screen(x, 0.0) 
            self.create_line(position[0], position[1], position[0], position[1] + 2,  fill=self.axiscolor)
            if self.ticklabel:
                self.create_text(position[0] + 10, position[1] + 5, text='%g' % (x), anchor=N) # shift it a little right and down to make the text visible
            x = x + self.xinterval
        y = self.axes.ymin
        while y < self.axes.ymax:
            position = self.axes.screen(0, y)
            self.create_line(position[0], position[1], position[0] - 2, position[1],  fill=self.axiscolor)
            if self.ticklabel:
                self.create_text(position[0] - 5, position[1] - 10, text='%g' % (y), anchor=E) # shift it a little up and left to make the text visible
            y = y + self.yinterval
            
    def _do_update(self, event):
        self.show()

class IzhikevichGUI:
    def __init__(self):
        self.demo = IzhikevichDemo()
        self.figureNums = {} # maps figure no. to label
        self.root = Tk()
        self.root.title('MOOSE - Izhikevich Demo')
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)
        self.controlPanel = Frame(self.root)
        self.buttons = {}
        for key in self.demo.parameters:
            button = Button(self.controlPanel, text=key)
            button.bind('<Button-1>', self._run_model)
            self.buttons[key] = button
            self.figureNums[IzhikevichDemo.parameters[key][0]] = key

        keys = self.figureNums.keys()
        keys.sort()
        ii = 0
        for key in keys:
            label = self.figureNums[key]
            button = self.buttons[label]
            button.grid(row=ii, column=0, sticky=W)
            button.pack(fill=BOTH, expand=1)
            ii = ii + 1
        self.controlPanel.grid(row=0, column=0, sticky=W)
        self.controlPanel.pack(side=LEFT)
        self.plotPanel = Frame(self.root, relief=RIDGE)   
        self.plot = Tkplot(self.plotPanel)
        self.plot.config(scrollregion=self.plot.bbox(ALL))
        self.plot.set_axes(-10, -10, 100, 100, 10, 10)
        self.plot.axes.xmin = -10
        self.plot.axes.ymin = -150
        self.plot.grid(row=0, column=1, sticky=S)
        self.plot.pack(fill=BOTH, expand=1)
        self.plotPanel.grid(row=0, rowspan=len(IzhikevichDemo.parameters), column=1, sticky=N+S+E+W)
        self.plotPanel.pack(fill=BOTH, side=RIGHT, expand=1)
        self.root.update()
        self.plot._do_update(None) # To force drawing the plot in the first place
        self.plot.bind('<Configure>', self.plot._do_update) # Connect the resize event to update of the plot

    def _run_model(self, event):
        key = event.widget['text']
        try:
            (time, Vm, Im) = self.demo.simulate(key)
        except NotImplementedError as (err, msg):
            tkMessageBox.showinfo('%s' % (err), '%s' % (msg))
            return
        length = len(time)
        self.plot.clear()
        self.plot.set_axes(-10, -150, time[length-1], 50, 10, 10)
        self.plot.plot(time, numpy.array(Vm) * 1e3)
        self.plot.plot(time, numpy.array(Im) * 1e9 - 100, color='red')
        self.plot.show()

    def start(self):
        self.root.mainloop()

if __name__ == '__main__':
    demo = IzhikevichGUI()
    demo.start()



# 
# demogui_tk.py ends here

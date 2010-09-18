#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Michaelis.py --- 
# 
# Filename: Michaelis.py
# Description: 
# Author: Gael Jalowicki
# Maintainer: 
# Created: Sat Jul 17 11:57:39 2010 (+0530)
# Version: 
# Last-Updated: Tue Jul 20 17:42:56 2010 (+0530)
#           By:
#     Update #:
# URL: 
# Keywords: 
# Compatibility: 
# 
# 

# Commentary: 
# 
# 
# 
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
# Implementation of the Michaelis Menten Kinetics


import sys
import os
import time

import pylab
import numpy
import logging

try:
    import moose
except ImportError:
    print "ERROR: Could not import moose. Please add the directory containing moose.py in your PYTHONPATH"
    sys.exit(1)

import pymoose

class Michaelis(object):

	def __init__(self):
		self.config()
		self.data = moose.Neutral('/data')
		self.paramDict = {}
		self.kincomp = moose.KinCompt('kincomp')
		self.kincomp.volume = self.VOL

		self.enzmol = moose.Molecule('enzmol', self.kincomp)
		self.enz = moose.Enzyme('enz', self.enzmol)
		self.sub = moose.Molecule('/kincomp/sub')
		self.prd = moose.Molecule('/kincomp/prd')

		self.sub.concInit = 1.0
		self.prd.concInit = 0.0
		self.enzmol.concInit = 1.0

		self.enz.k1 = 1.0
		self.enz.k2 = 4.0
		self.enz.k3 = 1.0

		self.enz.connect('sub', self.sub, 'reac')
		self.enz.connect('prd', self.prd, 'prd')
		self.enzmol.connect('reac',self.enz,'enz')

		self.enz.mode = 0

		self.cplx = moose.Molecule('/kincomp/enzmol/enz/enz_cplx')
		self.setClock()
		self.recording()


	def config(self):
		self.SIMDT = 1.0e-2
		self.PLOTDT = 1.0
		self.RUNTIME = 200.0
		# 1e3/1.66053878316e-21 = 6.022141790009886e+23; volume in Liters.
		self.VOL = 1.66053878316e-21
		self.context = moose.PyMooseBase.getContext()

	def recording(self):

		self.tabsub = moose.Table('/data/sub')
		self.tabsub.stepMode = 3
		self.tabsub.connect("inputRequest", self.sub, "conc")

		self.tabprd = moose.Table('/data/prd')
		self.tabprd.stepMode = 3
		self.tabprd.connect("inputRequest", self.prd, "conc")

		self.tabenzmol = moose.Table('/data/enzmol')
		self.tabenzmol.stepMode = 3
		self.tabenzmol.connect("inputRequest", self.enzmol, "conc")

		self.tabcomplx = moose.Table('/data/complx')
		self.tabcomplx.stepMode = 3
		self.tabcomplx.connect("inputRequest", self.cplx, "conc")		


	def calculatingV(self):
		"""Calculating and plotting the reaction velocity.
		"""
		self.vTable = moose.Table('/data/v')
		self.vTable.xmin = self.tabsub.xmin
		self.vTable.xmax = self.tabsub.xmax
		self.vTable.stepMode = 2
		self.vTable.xdivs = self.tabsub.xdivs
		index  = time = 0
		dt  = self.SIMDT
		for value in self.tabsub:
			self.vTable[index] = (self.enz.k3*self.enzmol.concInit*value*time)/(self.enz.Km+value)
			index += 1
			time += dt


	def calculatingComplex(self):
		"""This function is used by Qt Classes; it enables to dodge a bug when switching from implicit form to explicit form.
		"""
		print "in calculatingComplex"
		index = 0
		self.tabcomplx2 = moose.Table("/data/complx2")
		self.tabcomplx2.xmin = self.tabenzmol.xmin
		self.tabcomplx2.xmax = self.tabenzmol.xmax
		self.tabcomplx2.stepMode = 2
		self.tabcomplx2.xdivs = self.tabenzmol.xdivs
		for value in self.tabenzmol:
			self.tabcomplx2[index] = 1.0 - value
			index += 1
		self.tabcomplx2.stepMode = 3


	def setClock(self):
		self.context.setClock(0, self.SIMDT)
		self.context.setClock(1, self.SIMDT)
		self.context.setClock(2, self.PLOTDT)
		self.context.useClock(2, self.data.path + '/#[TYPE=Table]')


	def doRun(self):
		time_ = time.clock()
		self.context.step(self.RUNTIME)
		print "Running time: ", time.clock()-time_


	def plotting(self):
		pylab.plot(numpy.array(self.tabsub), label='Substrate')
		pylab.plot(numpy.array(self.tabprd), label='Product')
		pylab.plot(numpy.array(self.tabcomplx), label='complex')
		pylab.plot(numpy.array(self.tabenzmol), label='free enzyme')
		pylab.legend()
		pylab.show()


	def doResetForMichaelis(self):
		"""Here we reset the parameters of the model.
		"""
		self.doReset()
		self.SIMDT = self.paramDict["simDt"]
		self.PLOTDT = self.paramDict["plotDt"]
		self.RUNTIME = self.paramDict["runtime"]
		self.sub.concInit = self.paramDict["subConc"]
		self.prd.concInit = self.paramDict["prdConc"]
		self.enzmol.concInit = self.paramDict["enzConc"]
		self.VOL = self.paramDict["volume"]
		self.enz.Km = self.paramDict["Km"]
		self.enz.kcat = self.paramDict["k3"]
		self.enz.mode = self.paramDict["enzForm"]
		self.doReset()
		self.setClock()
		print self.paramDict


	def doReset(self):
		self.context.reset()

	def _tabsub(self):
		"""Get function used by Qt Classes.
		"""
		return self.tabsub

	def _tabprd(self):
		"""Get function used by Qt Classes.
		"""
		return self.tabprd

	def _tabcomplx(self):
		"""Get function used by Qt Classes.
		"""
		return self.tabcomplx

	def _tabcomplx2(self):
		"""Get function used by Qt Classes.
		"""
		return self.tabcomplx2

	def _tabenzmol(self):
		"""Get function used by Qt Classes.
		"""
		return self.tabenzmol

	def _tabv(self):
		"""vTable is only calculated after running. Get function used by Qt Classes.
		"""
		self.calculatingV()
		return self.vTable

	def _simDt(self):
		"""Get function used by Qt Classes.
		"""
		return self.SIMDT

	def _plotDt(self):
		"""Get function used by Qt Classes.
		"""
		return self.PLOTDT

	def _runtime(self):
		"""Get function used by Qt Classes.
		"""
		return self.RUNTIME

	def _volume(self):
		"""Get function used by Qt Classes.
		"""
		return self.kincomp.volume

	def _subConc(self):
		"""Get function used by Qt Classes.
		"""
		return self.sub.concInit

	def _prdConc(self):
		"""Get function used by Qt Classes.
		"""
		return self.prd.concInit

	def _enzConc(self):
		"""Get function used by Qt Classes.
		"""
		return self.enzmol.concInit

	def _kcat(self):
		"""Get function used by Qt Classes.
		"""
		return self.enz.kcat

	def _k1(self):
		"""Get function used by Qt Classes.
		"""
		return self.enz.k1

	def _k2(self):
		"""Get function used by Qt Classes.
		"""
		return self.enz.k2

	def _k3(self):
		"""Get function used by Qt Classes.
		"""
		return self.enz.k3

	# (6.02214179e23 * 1e-3 * self.kincomp.size) corresponds to the volScale constant in Enzyme.cpp
	def _Km(self):
		"""Return the non scaled value of Km.
		"""
		return self.enz.Km * 6.02214179e23 * 1e-3 * self.kincomp.size

	def _Vm(self):
		"""Get function used by Qt Classes.
		"""
		return self.enz.k3*(self.enzmol.concInit + self.cplx.concInit)

	def _ratio(self):
		"""Get function used by Qt Classes.
		"""
		return self.enz.k2/self.enz.k3

	def dumpPlotData(self):
		self.tabsub.dumpFile("michaelisSubstrate.plot")
		self.tabprd.dumpFile("michaelisProduct.plot")
		print "Plots in michaelisSubstrate.plot and in michaelisProduct.plot"

def testMM():
	mm = Michaelis()
	mm.doReset()
	mm.doRun()
	mm.calculatingV()
	mm.plotting()

if __name__ == '__main__':
	testMM()



#
# Michaelis.py ends here

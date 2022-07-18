# -*- coding: utf-8 -*-
# test_reader.py ---
#
# Filename: test_reader3.py
# Description:
# Author: Anal Kumar
# Maintainer:
# Version:
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

# Code:

from __future__ import print_function
import unittest
import moose
from reader import NML2Reader
import neuroml as nml
import os
import numpy as np

'''
Manually calculated the Vm curve given the parameters in the test_files/passiveCell.nml file. Based on the curve, the follwing are tested:
first element should be almostequal to -66.6mV
Average of the last 100ms should  be -54.3mV
Ra should be 21409.5
steady state value of Vm between 0.090 to 0.100s is -0.02761+-0.0002
Vm at 0.0533s is -0.03725 (at 1 tau time)
'''

class TestPassiveCell(unittest.TestCase):
    def setUp(self):
        if '/library' in moose.le():
            moose.delete('/library')
        self.reader = NML2Reader(verbose=True)
        self.lib = moose.Neutral('/library')
        self.filename = os.path.realpath('test_files/passiveCell.nml')
        self.reader.read(self.filename)
        self.mcell = moose.element('/library/pop0/0')
        self.soma = moose.element(self.mcell.path + '/soma')
        if not moose.exists('vmtab'):
            self.vmtab = moose.Table('vmtab')
            moose.connect(self.vmtab, 'requestOut', self.soma, 'getVm')
        moose.reinit()
        moose.start(300e-3)
        self.t = np.linspace(0, 300e-3, len(self.vmtab.vector))

    def test_currentinj(self):
        self.assertAlmostEqual(self.vmtab.vector[0], -66.6e-3, delta=0.5e-3)
        self.assertAlmostEqual(np.mean(self.vmtab.vector[self.t>=200e-3]), -54.3e-3, delta=0.5e-3)
        self.assertAlmostEqual(self.soma.Ra, 21409.5, delta=500)
        self.assertAlmostEqual(np.mean(self.vmtab.vector[(self.t>=90e-3) & (self.t<100e-3)]), -0.0276, delta=0.5e-3)
        self.assertAlmostEqual(self.vmtab.vector[np.argmin(np.abs(self.t-0.0533))], -0.03725, delta=0.5e-3)


if __name__ == '__main__':
    unittest.main()
    #p = TestPassiveCell()

#
# test_reader.py ends here

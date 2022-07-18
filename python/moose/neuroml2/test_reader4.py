# -*- coding: utf-8 -*-
# test_reader.py ---
#
# Filename: test_reader.py
# Description:
# Author:
# Maintainer:
# Created: Wed Jul 24 16:02:21 2013 (+0530)
# Version:
# Last-Updated: Sun Apr 17 16:13:01 2016 (-0400)
#           By: subha
#     Update #: 112
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
import numpy as np
import moose
import neuroml as nml
from reader import NML2Reader
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

class TestFullCell(unittest.TestCase):
    def setUp(self):
        if '/library' in moose.le():
            moose.delete('/library')
        self.reader = NML2Reader(verbose=True)

        self.lib = moose.Neutral('/library')
        self.filename = os.path.realpath('test_files/NML2_SingleCompHHCell.nml')
        self.reader.read(self.filename)
        self.soma = moose.element('library/hhpop/0/soma')

        if not moose.exists('vmtab'):
            self.vmtab = moose.Table('vmtab')
            moose.connect(self.vmtab, 'requestOut', self.soma, 'getVm')
        moose.reinit()
        moose.start(300e-3)
        self.t = np.linspace(0, 300e-3, len(self.vmtab.vector))

    def test_currentinj(self):
        # self.assertAlmostEqual(self.vmtab.vector[0], -66.6e-3, delta=0.5e-3)
        # self.assertAlmostEqual(np.mean(self.vmtab.vector[self.t>=200e-3]), -54.3e-3, delta=0.5e-3)
        # self.assertAlmostEqual(self.soma.Ra, 21409.5, delta=500)
        # self.assertAlmostEqual(np.mean(self.vmtab.vector[(self.t>=90e-3) & (self.t<100e-3)]), -0.0276, delta=0.5e-3)
        # self.assertAlmostEqual(self.vmtab.vector[np.argmin(np.abs(self.t-0.0533))], -0.03725, delta=0.5e-3)
        peaks, peaks_dict = find_peaks(self.vmtab.vector, height=0)
        act_peaks = [0.1024, 0.1186, 0.1346, 0.1507, 0.1667, 0.1827, 0.1987]
        act_peaks_height = [0.03985157, 0.03131845, 0.03058362, 0.0305823 , 0.03078506, 0.03095396, 0.03086463]
        self.assertEqual(len(peaks), len(act_peaks))
        for i in range(len(act_peaks)):
            self.assertAlmostEqual(self.t[peaks][i], act_peaks[i], delta=1e-3)
            self.assertAlmostEqual(self.vmtab.vector[peaks][i], act_peaks_height[i], delta=5e-3)

if __name__ == '__main__':
    unittest.main()

#
# test_reader.py ends here

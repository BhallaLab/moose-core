# test_calc.py --- 
# 
# Filename: test_calc.py
# Description: 
# Author: 
# Maintainer: 
# Created: Tue May 28 09:28:13 2013 (+0530)
# Version: 
# Last-Updated: Tue May 28 09:33:24 2013 (+0530)
#           By: subha
#     Update #: 18
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

import unittest
import moose

class TestCalc(unittest.TestCase):
    def setUp(self):
        self.calc = moose.Calc('/test_calc')

    def testMultiVarAvg(self):
        n = 5
        expr = 'avg('
        for ii in range(n-1):
            expr += ' x_%d,' % (ii)
        expr += 'x_%d )' % (ii)
        self.calc.expr = expr
        self.calc.mode = 3
        for ii in range(n):
            self.calc.var['x_%d' % (n)] = 1.0
        self.assertAlmostEqual(self.calc.value, 1.0)

if __name__ == '__main__':
    unittest.main()

# 
# test_calc.py ends here

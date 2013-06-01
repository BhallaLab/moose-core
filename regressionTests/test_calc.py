# test_calc.py --- 
# 
# Filename: test_calc.py
# Description: 
# Author: 
# Maintainer: 
# Created: Tue May 28 09:28:13 2013 (+0530)
# Version: 
# Last-Updated: Sat Jun  1 18:50:30 2013 (+0530)
#           By: subha
#     Update #: 100
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
from datetime import datetime

class TestCalc(unittest.TestCase):
    def setUp(self):
        self.calc = moose.Calc('/test_calc')

    def testMultiVarAvg(self):
        n = 5
        expr = 'avg('
        for ii in range(n):
            expr += ' x_%d,' % (ii)        
        expr += 'x_%d )' % (n)
        print expr
        self.calc.expr = expr
        print self.calc.expr
        # self.calc.mode = 3
        for ii in range(n+1):
            self.calc.var['x_%d' % (ii)] = 1.0
        v = self.calc.value
        self.assertAlmostEqual(v, 1.0)

    def testParseSpeed(self):
        n = 100
        nv = 1000
        expr = ''
        for ii in range(nv):
            expr += 'x_%d * x_%d + ' % (ii, ii+1)
        expr += 'x_%d' % (nv)
        ts = datetime.now()
        for ii in range(n):
            self.calc.expr = expr
        te = datetime.now()
        td = te - ts
        print 'Time to parse %d-variable expression: %g s' % (nv, (td.seconds + 1e-6 * td.microseconds) / n)

    def testEvalSpeed(self):
        n = 1000
        nv = 100
        expr = ''
        for ii in range(nv):
            expr += 'x_%d * x_%d + ' % (ii, ii+1)
        expr += 'x_%d' % (nv)
        self.calc.expr = expr
        for ii in range(nv+1):
            self.calc.var['x_%d' % (ii)] = 1
        ts = datetime.now()
        for ii in range(n):
            v = self.calc.value        
        te = datetime.now()
        td = te - ts
        print 'Time to evaluate %d multiplications and %d additions: %g s' % (nv, nv+1, (td.seconds + 1e-6 * td.microseconds)/n)
        

if __name__ == '__main__':
    unittest.main()

# 
# test_calc.py ends here

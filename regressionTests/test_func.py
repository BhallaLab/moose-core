# test_func.py --- 
# 
# Filename: test_func.py
# Description: 
# Author: 
# Maintainer: 
# Created: Tue May 28 09:28:13 2013 (+0530)
# Version: 
# Last-Updated: Sat Jun  1 19:06:54 2013 (+0530)
#           By: subha
#     Update #: 102
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

class TestFunc(unittest.TestCase):
    def setUp(self):
        self.func = moose.Func('/test_func')

    def testMultiVarAvg(self):
        n = 5
        expr = 'avg('
        for ii in range(n):
            expr += ' x_%d,' % (ii)        
        expr += 'x_%d )' % (n)
        print expr
        self.func.expr = expr
        print self.func.expr
        # self.func.mode = 3
        for ii in range(n+1):
            self.func.var['x_%d' % (ii)] = 1.0
        v = self.func.value
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
            self.func.expr = expr
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
        self.func.expr = expr
        for ii in range(nv+1):
            self.func.var['x_%d' % (ii)] = 1
        ts = datetime.now()
        for ii in range(n):
            v = self.func.value        
        te = datetime.now()
        td = te - ts
        print 'Time to evaluate %d multiplications and %d additions: %g s' % (nv, nv+1, (td.seconds + 1e-6 * td.microseconds)/n)
        

if __name__ == '__main__':
    unittest.main()

# 
# test_func.py ends here

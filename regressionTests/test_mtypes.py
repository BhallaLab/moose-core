# test_mtypes.py --- 
# 
# Filename: test_mtypes.py
# Description: 
# Author: 
# Maintainer: 
# Created: Fri Feb  8 11:31:58 2013 (+0530)
# Version: 
# Last-Updated: Fri Feb  8 12:52:43 2013 (+0530)
#           By: subha
#     Update #: 19
# URL: 
# Keywords: 
# Compatibility: 
# 
# 

# Commentary: 
# 
# Tests for mtypes utility
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
from moose.mtypes import isKKIT

class IsKKITTestCase(unittest.TestCase):
    def setUp(self):
        print '--------------------------------------------------------------------'
    def testComment(self):
        self.assertTrue(isKKIT('mtypes/iskkit_testitem1.g'))

    def testMultilineComment(self):
        self.assertTrue(isKKIT('mtypes/iskkit_testitem2.g'))

    def testMultilineCommentWithLeadingGarbage(self):
        self.assertTrue(isKKIT('mtypes/iskkit_testitem3.g'))

    def testCPPCommentedKKIT(self):
        self.assertFalse(isKKIT('mtypes/iskkit_testitem4.g'))
        
    def testCCommentedKKIT(self):
        self.assertFalse(isKKIT('mtypes/iskkit_testitem5.g'))

    def testMultilineGarbage(self):
        self.assertTrue(isKKIT('mtypes/iskkit_testitem6.g'))

if __name__ == '__main__':
    unittest.main()

# 
# test_mtypes.py ends here

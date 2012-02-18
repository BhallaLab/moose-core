# test_kkit.py --- 
# 
# Filename: test_kkit.py
# Description: 
# Author: Subhasis Ray
# Maintainer: 
# Created: Sat Feb 18 16:56:21 2012 (+0530)
# Version: 
# Last-Updated: Sat Feb 18 17:10:27 2012 (+0530)
#           By: Subhasis Ray
#     Update #: 17
# URL: 
# Keywords: 
# Compatibility: 
# 
# 

# Commentary: 
# 
# Test Kinetikit via Python.
# 
# 

# Change log:
# 
# 2012-02-18 16:57:01 (+0530) - Subha: Initial version based on code
# provided by Harsha.
# 

# Code:

import unittest
import moose

class TestKkit(unittest.TestCase):
    def test_loadModel(self):
        filename = 'acc8.g'
        target = '/acc8'
        loaded_id = moose.loadModel(filename, target)
        self.assertEqual(target, loaded_id.path)
        kinetics_path = target + '/kinetics'
        self.assertTrue(moose.exists(kinetics_path))
        # Check BUG: Assertion failure in Neutral::path on Zombie
        # objects - ID: 3488106
        child_paths = []
        for child in moose.Neutral(kinetics_path).children:
            child_paths.append(child.getPath())
    
if __name__ == '__main__':
    unittest.main()
# 
# test_kkit.py ends here

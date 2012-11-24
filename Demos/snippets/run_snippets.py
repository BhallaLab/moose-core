#!/usr/bin/env python
# runall.py --- 
# 
# Filename: runall.py
# Description: 
# Author: 
# Maintainer: 
# Created: Sat Nov 24 19:10:14 2012 (+0530)
# Version: 
# Last-Updated: Sat Nov 24 19:47:51 2012 (+0530)
#           By: subha
#     Update #: 37
# URL: 
# Keywords: 
# Compatibility: 
# 
# 

# Commentary: 
# 
# Run all snippets in this directory.
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

"""Run all snippets in the current directory.

Spawns new Python process to run the scripts.
"""
import os
import sys
from subprocess import call

os.environ['NUMPTHREADS'] = '1'
os.environ['PYTHONPATH'] = os.environ['PYTHONPATH'] + ':../../python'

thisfile = os.path.abspath(__file__)
def get_snippet_files(directory='.'):
    snippets = set()
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith('.py'):
                filepath = os.path.abspath(os.path.join(root, filename))
                snippets.add(filepath)
    try:
        snippets.remove(thisfile)
    except KeyError:
        pass
    return snippets

def run_snippets(files):
    for filepath in files:
        print 'Running', filepath, '...',
        ret = call(['python', filepath])
        if ret:
            print 'ERROR: exit code:', ret
        else:
            print 'OK'
    
if __name__ == '__main__':
    files = get_snippet_files()
    run_snippets(files)
    
    


# 
# runall.py ends here

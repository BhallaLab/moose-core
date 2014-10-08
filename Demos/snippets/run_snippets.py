#!/usr/bin/env python
# runall.py --- 
# 
# Filename: runall.py
# Description: 
# Author: 
# Maintainer: 
# Created: Sat Nov 24 19:10:14 2012 (+0530)
# Version: 
# Last-Updated: Mon Nov 26 11:03:49 2012 (+0530)
#           By: subha
#     Update #: 51
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

NOTE: Some of the snippets use matplotlib to display interactive plots
(so the excecution will stop unltil you manually close the plot). In
order to run this non-interactively, you can edit your matplotlibrc
(usually located in ~/.matplotlib/) and add the following line:

backend: Agg

"""
import os
import sys
from subprocess import call

os.environ['NUMPTHREADS'] = '1'
sys.path.append('../../python')
#os.environ['PYTHONPATH'] = os.environ['PYTHONPATH'] + ':../../python'

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
    # First change to current directory
    if len(sys.argv) == 1:
        os.chdir(os.path.dirname(thisfile))
    else:
        try:
            os.chdir(sys.argv[1])
        except OSError:
            print """Usage: %s [directory]\n
Exceute each python file in directory one by one in separate Python 
subprocess. If directory is not specified, location of this file is 
used.""" % (sys.argv[0])
            sys.exit(1)
    files = get_snippet_files()
    run_snippets(files)
    
    


# 
# runall.py ends here

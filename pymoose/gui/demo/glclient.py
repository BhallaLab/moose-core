# glclient.py --- 
# 
# Filename: glclient.py
# Description: 
# Author: Subhasis Ray
# Maintainer: 
# Created: Thu Feb  4 14:52:22 2010 (+0530)
# Version: 
# Last-Updated: Thu Feb  4 22:06:37 2010 (+0530)
#           By: Subhasis Ray
#     Update #: 105
# URL: 
# Keywords: 
# Compatibility: 
# 
# 

# Commentary: 
# 
# This is a wrapper arounf glclient program for 3D visualization in MOOSE.
# 34
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

import sys
from getopt import getopt
from subprocess import Popen

GLCLIENT_EXE = '../../../gl/src/glclient'

option_dict={'-p': '9999',
	     '-m': 'c',
	     '-c': '../../../gl/colormaps/rainbow2' # This must be updated with a relative path
	     }


class GLClient(object):
    def __init__(self, port='9999', mode='c', colormap='../../../gl/colormaps/rainbow2'):
	self.opt_dict = {}
	self.opt_dict['-p'] = port
	self.opt_dict['-m'] = mode
	self.opt_dict['-c'] = colormap
        self.run()
    def run(self, option_dict=None):
	cmd = [GLCLIENT_EXE]
	if option_dict:
	    for key, value in option_dict.items():
		value = option_dict[key]
		self.opt_dict[key] = value
	
	for key, value in self.opt_dict.items():
	    cmd.extend([key, value])

	child = Popen(cmd)
	return child

if __name__ == '__main__':
    print 'sys.argv:', sys.argv
    options,args = getopt(sys.argv[1:], 'p:c:m:d:a:')
    opt_dict = dict(options)
    client = GLClient()
    child = client.run()
    


# 
# glclient.py ends here

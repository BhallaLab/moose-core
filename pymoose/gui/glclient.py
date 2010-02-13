# glclient.py --- 
# 
# Filename: glclient.py
# Description: 
# Author: Subhasis Ray
# Maintainer: 
# Created: Thu Feb  4 14:52:22 2010 (+0530)
# Version: 
# Last-Updated: Sat Feb 13 20:38:12 2010 (+0530)
#           By: Subhasis Ray
#     Update #: 129
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

GLCLIENT_EXE = '../../gl/src/glclient'

option_dict={'-p': '9999',
	     '-m': 'c',
	     '-c': '../../../gl/colormaps/rainbow2' # This must be updated with a relative path
	     }


class GLClient(object):
    def __init__(self, exe=GLCLIENT_EXE, port='9999', mode='c', colormap='../../../gl/colormaps/rainbow2'):
	self.opt_dict = {}
	self.opt_dict['-p'] = port
	self.opt_dict['-m'] = mode
	self.opt_dict['-c'] = colormap
        self.executable = exe
        self.running = False
        self.run(self.opt_dict)

    def run(self, option_dict=None):
	cmd = [self.executable]
	if option_dict:
	    for key, value in option_dict.items():
		value = option_dict[key]
		self.opt_dict[key] = value
	
	for key, value in self.opt_dict.items():
	    cmd.extend([key, value])
        if self.running:
            self.stop()
        self.running = True
	self.child = Popen(cmd)
	return self.child

    def stop(self):
        if not self.running:
            print self.__class__.__name__, ': stop() - client is not running'
            return
        self.child.kill()
        self.child.wait()
        self.running = False

if __name__ == '__main__':
    print 'sys.argv:', sys.argv
    options,args = getopt(sys.argv[1:], 'p:c:m:d:a:')
    opt_dict = dict(options)
    client = GLClient()
    child = client.run()
    


# 
# glclient.py ends here

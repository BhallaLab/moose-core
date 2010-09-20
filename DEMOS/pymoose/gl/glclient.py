# glclient.py --- 
# 
# Filename: glclient.py
# Description: 
# Author: Subhasis Ray
# Maintainer: 
# Created: Thu Feb  4 14:52:22 2010 (+0530)
# Version: 
# Last-Updated: Mon Sep 20 15:44:54 2010 (+0530)
#           By: Subhasis Ray
#     Update #: 157
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
import os
import sys
from getopt import getopt
from subprocess import Popen

import config

option_dict={'-p': '9999',
	     '-m': 'c',
	     '-c': os.path.join('colormaps', 'rainbow2') # This must be updated with a relative path
	     }


class GLClient(object):
    def __init__(self, exe=None, port=None, mode='c', colormap=None):
        if exe is None:
            exe = config.get_settings().value(config.KEY_GL_CLIENT_EXECUTABLE).toString()
        if port is None:
            port = config.get_settings().value(config.KEY_GL_PORT).toString()
        if colormap is None:
            colormap = config.get_settings().value(config.KEY_GL_COLORMAP).toString()
	self.opt_dict = {}
	self.opt_dict['-p'] = str(port)
	self.opt_dict['-m'] = str(mode)
	self.opt_dict['-c'] = str(colormap)
        self.executable = str(exe)
        self.running = False
        self.run()

    def run(self, option_dict=None):
	cmd = [self.executable]
	if option_dict:
	    for key, value in option_dict.items():
		value = option_dict[key]
		self.opt_dict[key] = value
	
	for key, value in self.opt_dict.items():
	    cmd.extend([key, value])
            self.stop()
        self.running = True
        print 'Executing client with the following command line:'
        for item in cmd: print item, 
        print
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

#*******************************************************************
# File:            squid_func.py
# Description:      
# Author:          Subhasis Ray
# E-mail:          ray.subhasis@gmail.com
# Created:         2008-01-17 14:40:13
#*******************************************************************/
#**********************************************************************
#* This program is part of 'MOOSE', the
#* Messaging Object Oriented Simulation Environment,
#* also known as GENESIS 3 base code.
#*           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
#* It is made available under the terms of the
#* GNU General Public License version 2
#* See the file COPYING.LIB for the full notice.
#*********************************************************************/

def squid_compt(path, length, diameter):
    if not instanceof(path, string):
        print "Error: path must be string"
        exit(0)
    else if not instanceof(length, double):
        print "Error: length must be a numebr"
        exit(0)
    else if not instanceof(diameter, double):
        print "Error: diameter must be a number"

    cmpt = moose.Compartment(path)
    cmpt.length, cmpt.diameter, cmpt.angle  = length, diameter, 90
    
    # TODO: complete
# cmpt.getContext().do_deep_copy(PyMooseBase.pathToId(
    

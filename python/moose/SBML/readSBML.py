'''
*******************************************************************
 * File:            readSBML.py
 * Description:
 * Author:          HarshaRani
 * E-mail:          hrani@ncbs.res.in
 ********************************************************************/
/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2016 Upinder S. Bhalla. and NCBS
Created : Thu May 12 10:19:00 2016(+0530)
Version 
Last-Updated:
		  By:
**********************************************************************/
/****************************

'''
import sys
import os.path
import collections
from moose import *
import libsbml

def mooseReadSBML(filepath,loadpath):
	print " filepath ",filepath
	document = libsbml.readSBML(filepath)
	if (document.getNumErrors() > 0):
		print("Encountered the following SBML errors:" )
		document.printErrors()
		return False;
	else:
		level = document.getLevel();
		version = document.getVersion();
		print("\n" + "File: " + filepath + " (Level " + str(level) + ", version " + str(version) + ")" );
		#model = document.getModel();
	return loadpath



if __name__ == "__main__":
    
    filepath = sys.argv[1]
    path = sys.argv[2]
    
    f = open(filepath, 'r')
    
    if path == '':
    	loadpath = filepath[filepath.rfind('/'):filepath.find('.')]
    else:
    	loadpath = path
    
    read = mooseReadSBML(filepath,loadpath)
    if read:
        print " Read to path",loadpath
    else:
        print " could not read  SBML to MOOSE"

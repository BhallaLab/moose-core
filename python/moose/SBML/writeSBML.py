'''
*******************************************************************
 * File:            writeSBML.py
 * Description:
 * Author:          HarshaRani
 * E-mail:          hrani@ncbs.res.in
 ********************************************************************/
/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2016 Upinder S. Bhalla. and NCBS
Created : Friday May 27 12:19:00 2016(+0530)
Version 
Last-Updated:
		  By:
**********************************************************************/
/****************************

'''
from moose import *
from libsbml import *
import re
from collections import Counter
#from moose import wildcardFind, element, loadModel, ChemCompt, exists, Annotator, Pool, ZombiePool,PoolBase,CplxEnzBase,Function,ZombieFunction


def mooseWriteSBML(modelpath,filename):
    sbmlDoc = SBMLDocument(3, 1)
    filename1 = filename[:filename.find('.')]
    filename = filename1+".xml"
    print " filename" ,modelpath,	filename
    return filename

if __name__ == "__main__":

	filepath = sys.argv[1]
	path = sys.argv[2]

	f = open(filepath, 'r')
	
	if path == '':
		loadpath = filepath[filepath.rfind('/'):filepath.find('.')]
	else:
		loadpath = path
	
	moose.loadModel(filepath,loadpath,"gsl")
	
	written = mooseWriteSBML(loadpath,filepath)
	if written:
	    print " File written to ",written
	else:
		print " could not write model to SBML file"
	
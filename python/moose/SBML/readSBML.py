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

'''
   TODO in
    -Compartment
      --Need to add group
      --Need to deal with compartment outside
    -Molecule
      -- Need to add group 
      -- Func pool and its math calculation need to be added.
    -Loading Model from SBML
      --Tested 1-30 testcase example model provided by l3v1 and l2v4 std.
        ---These are the models that worked (sbml testcase)1-6,10,14-15,17-21,23-25,34,35,58
	---Need to check
	 	----what to do when boundarycondition is true i.e.,
             differential equation derived from the reaction definitions
             should not be calculated for the species(7-9,11-13,16)
        	 ----kineticsLaw, Math fun has fraction,ceiling,reminder,power 28etc.
         	----Events to be added 26
	 	----initial Assisgment for compartment 27
        	 ----when stoichiometry is rational number 22
	 	---- For Michaelis Menten kinetics km is not defined which is most of the case need to calculate
'''

def mooseReadSBML(filepath,loadpath):
	print " filepath ",filepath
	try:
		filep = open(filepath, "r")
		document = libsbml.readSBML(filepath)
		num_errors = document.getNumErrors()
		if ( num_errors > 0 ):
			print("Encountered the following SBML errors:" );
			document.printErrors();
			return False;
		else:
			level = document.getLevel();
			version = document.getVersion();
			print("\n" + "File: " + filepath + " (Level " + str(level) + ", version " + str(version) + ")" );
			model = document.getModel();
			if (model == None):
				print("No model present." );
				return False;
			else:
				print " model ",model
				print("functionDefinitions: " + str(model.getNumFunctionDefinitions()) );
				print("    unitDefinitions: " + str(model.getNumUnitDefinitions()) );
				print("   compartmentTypes: " + str(model.getNumCompartmentTypes()) );
				print("        specieTypes: " + str(model.getNumSpeciesTypes()) );
				print("       compartments: " + str(model.getNumCompartments()) );
				print("            species: " + str(model.getNumSpecies()) );
				print("         parameters: " + str(model.getNumParameters()) );
				print(" initialAssignments: " + str(model.getNumInitialAssignments()) );
				print("              rules: " + str(model.getNumRules()) );
				print("        constraints: " + str(model.getNumConstraints()) );
				print("          reactions: " + str(model.getNumReactions()) );
				print("             events: " + str(model.getNumEvents()) );
				print("\n");

	except IOError:
		print "File " ,filepath ," does not exist."


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

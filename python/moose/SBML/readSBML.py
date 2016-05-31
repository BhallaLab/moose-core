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

				if (model.getNumCompartments() == 0):
					return False
				else:
					baseId = moose.Neutral(loadpath)
					#All the model will be created under model as a thumbrule
					basePath = moose.Neutral(baseId.path+'/model')
					#Map Compartment's SBML id as key and value is list of[ Moose ID and SpatialDimensions ]
					comptSbmlidMooseIdMap = {}
					print ": ",basePath.path
					globparameterIdValue = {}
					modelAnnotaInfo = {}
					mapParameter(model,globparameterIdValue)
					errorFlag = createCompartment(basePath,model,comptSbmlidMooseIdMap)

	except IOError:
		print "File " ,filepath ," does not exist."
		
def transformUnit(unitForObject):
	#print "unit ",UnitDefinition.printUnits(unitForObject.getDerivedUnitDefinition())
	lvalue = 1.0
	unitset = False
	if (unitForObject.getDerivedUnitDefinition()):
		unit = (unitForObject.getDerivedUnitDefinition())
		if not (unit.getNumUnits()):
			#If units are not defined then assume for
			# -- Compartment litre then m3 for moose
			# -- species mole then millimole for concentration
			# since both are needs to multiplied by 10-3 returning lvalue
			lvalue *= 0.001
        else:
			for ui in range(0,unit.getNumUnits()):
				unitType =  unit.getUnit(ui)
				if( unitType.isLitre()):
					exponent = unitType.getExponent()
					multiplier = unitType.getMultiplier()
					scale = unitType.getScale()
					offset = unitType.getOffset()
					#units for compartment is Litre but MOOSE compartment is m3
					scale = scale-3
					lvalue *= pow( multiplier * pow(10.0,scale), exponent ) + offset;
					unitset = True			
				if( unitType.isMole()):
					exponent = unitType.getExponent()
					multiplier = unitType.getMultiplier()
					scale = unitType.getScale()
					offset = unitType.getOffset()
					lvalue *= pow( multiplier * pow(10.0,scale), exponent ) + offset;
					unitset = True
	return (lvalue,unitset)
def createCompartment(basePath,model,comptSbmlidMooseIdMap):
	#ToDoList : Check what should be done for the spaitialdimension is 2 or 1, area or length
	if not(model.getNumCompartments()):
		return False
	else:
		for c in range(0,model.getNumCompartments()):
			compt = model.getCompartment(c)
			# print("Compartment " + str(c) + ": "+ UnitDefinition.printUnits(compt.getDerivedUnitDefinition()))
			msize = 0.0
			unitfactor = 1.0
			sbmlCmptId = None
			name = None

			if ( compt.isSetId() ):
				sbmlCmptId = compt.getId()
				
			if ( compt.isSetName() ):
				name = compt.getName()
				name = name.replace(" ","_space")
					
			if ( compt.isSetOutside() ):
				outside = compt.getOutside()
					
			if ( compt.isSetSize() ):
				msize = compt.getSize()
				if msize == 1:
					print "Compartment size is 1"

			dimension = compt.getSpatialDimensions();
			if dimension == 3:
				unitfactor,unitset = transformUnit(compt)
				
			else:
				print " Currently we don't deal with spatial Dimension less than 3 and unit's area or length"
				return False

			if not( name ):
				name = sbmlCmptId
			
			mooseCmptId = moose.CubeMesh(basePath.path+'/'+name)
			mooseCmptId.volume = (msize*unitfactor)
			comptSbmlidMooseIdMap[sbmlCmptId]={"MooseId": mooseCmptId, "spatialDim":dimension, "size" : msize}
	return True
def mapParameter(model,globparameterIdValue):
	for pm in range(0,model.getNumParameters()):
		prm = model.getParameter( pm );
		if ( prm.isSetId() ):
			parid = prm.getId()
		value = 0.0;
		if ( prm.isSetValue() ):
			value = prm.getValue()
		globparameterIdValue[parid] = value

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

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
      -- mathML only AssisgmentRule is taken partly I have checked addition and multiplication, 
       --, need to do for other calculation.
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
					if errorFlag:
						specInfoMap = {}
						errorFlag = createSpecies(basePath,model,comptSbmlidMooseIdMap,specInfoMap)
						if errorFlag:
							errorFlag = createRules(model,specInfoMap,globparameterIdValue)

					if not errorFlag:
						print " errorFlag ",errorFlag
						#Any time in the middle if SBML does not read then I delete everything from model level
						#This is important as while reading in GUI the model will show up untill built which is not correct
						print "Deleted rest of the model"
						moose.delete(basePath)
				return baseId;


	except IOError:
		print "File " ,filepath ," does not exist."

def getMembers(node,ruleMemlist):
	if node.getType() == libsbml.AST_PLUS:
		if node.getNumChildren() == 0:
			print ("0")
			return
		getMembers(node.getChild(0),ruleMemlist)
		for i in range(1,node.getNumChildren()):
			# addition
			getMembers(node.getChild(i),ruleMemlist)
	elif node.getType() == libsbml.AST_REAL:
		#This will be constant
		pass
	elif node.getType() == libsbml.AST_NAME:
		#This will be the ci term"
		ruleMemlist.append(node.getName())

	elif node.getType() == libsbml.AST_TIMES:
		if node.getNumChildren() == 0:
			print ("0")
			return
		getMembers(node.getChild(0),ruleMemlist)
		for i in range(1,node.getNumChildren()):
			# Multiplication
			getMembers(node.getChild(i),ruleMemlist)
	else:
		print " this case need to be handled"

def createRules(model,specInfoMap,globparameterIdValue):
	for r in range(0,model.getNumRules()):
			rule = model.getRule(r)
			if (rule.isAssignment()):
				rule_variable = rule.getVariable();
				poolList = specInfoMap[rule_variable]["Mpath"].path
				funcId = moose.Function(poolList+'/func')
				moose.connect( funcId, 'valueOut', poolList ,'setN' )
				ruleMath = rule.getMath()
				ruleMemlist = []
				speFunXterm = {}
				getMembers(ruleMath,ruleMemlist)
				for i in ruleMemlist:
					if (specInfoMap.has_key(i)):
						specMapList = specInfoMap[i]["Mpath"]
						numVars = funcId.numVars
						x = funcId.path+'/x['+str(numVars)+']'
						speFunXterm[i] = 'x'+str(numVars)
						moose.connect(specMapList , 'nOut', x, 'input' )
						funcId.numVars = numVars +1
					elif not(globparameterIdValue.has_key(i)):
						print "check the variable type ",i

				exp = rule.getFormula()
				for mem in ruleMemlist:
					if ( specInfoMap.has_key(mem)):
						exp1 = exp.replace(mem,str(speFunXterm[mem]))
						exp = exp1
					elif( globparameterIdValue.has_key(mem)):
						exp1 = exp.replace(mem,str(globparameterIdValue[mem]))
						exp = exp1
					else:
						print "Math expression need to be checked"
				funcId.expr = exp.strip(" \t\n\r")
				return True

			elif( rule.isRate() ):
				print "Warning : For now this \"",rule.getVariable(), "\" rate Rule is not handled in moose "
				return False

			elif ( rule.isAlgebraic() ):
				print "Warning: For now this " ,rule.getVariable()," Algebraic Rule is not handled in moose"
				return False
	return True

def pullnotes(sbmlId,mooseId):
	if sbmlId.getNotes() != None:
		tnodec = ((sbmlId.getNotes()).getChild(0)).getChild(0)
		notes = tnodec.getCharacters()
		notes = notes.strip(' \t\n\r')
		objPath = mooseId.path+"/info"
		if not moose.exists(objPath):
			objInfo = moose.Annotator(mooseId.path+'/info')
		else:
			objInfo = moose.element(mooseId.path+'/info')
		objInfo.notes = notes

def createSpecies(basePath,model,comptSbmlidMooseIdMap,specInfoMap):
	# ToDo:
	# - Need to add group name if exist in pool
	# - Notes
	# print "species "
	if not 	(model.getNumSpecies()):
		return False
	else:
		for sindex in range(0,model.getNumSpecies()):
			spe = model.getSpecies(sindex)
			sName = None
			sId = spe.getId()

			if spe.isSetName():
				sName = spe.getName()
				sName = sName.replace(" ","_space_")

			if spe.isSetCompartment():
				comptId = spe.getCompartment()

			if not( sName ):
				sName = sId

			constant = spe.getConstant()
			boundaryCondition = spe.getBoundaryCondition()
			comptEl = comptSbmlidMooseIdMap[comptId]["MooseId"].path
			hasonlySubUnit = spe.getHasOnlySubstanceUnits();
			# "false": is {unit of amount}/{unit of size} (i.e., concentration or density). 
			# "true": then the value is interpreted as having a unit of amount only.

			if (boundaryCondition):
				poolId = moose.BufPool(comptEl+'/'+sName)
			else:
				poolId = moose.Pool(comptEl+'/'+sName)
			
			if (spe.isSetNotes):
				pullnotes(spe,poolId)
					
			specInfoMap[sId] = {"Mpath" : poolId, "const" : constant, "bcondition" : boundaryCondition, "hassubunit" : hasonlySubUnit, "comptId" : comptSbmlidMooseIdMap[comptId]["MooseId"]}
			initvalue = 0.0
			unitfactor,unitset = transformUnit(spe)
			if(spe.isSetInitialAmount()):
				initvalue = spe.getInitialAmount()
				# moose is capable of populating number nInit so
				# checking hasonlySubstanceUnit doesn't matter,
				# populating nInit with this value automatically calculate the concInit.
				# default unit is mole to convert to number we need to multiply by
				# pow(NA) the avogadro's number
				if not (unitset):
					#unit is not set then pass milli mole which is not true for number
					# so setting unitfactor to 1
					unitfactor = 1
				initvalue = initvalue * unitfactor * pow(6.0221409e23,1);
				poolId.nInit = initvalue
			elif ( spe.isSetInitialConcentration() ):
				#ToDo : check 00976

				initvalue = spe.getInitialConcentration();
				#transValue will take care of multiplying any units are defined else millimole
				#print " initvalue ",initvalue, unitfactor
				unitfactor = 1
				initvalue = initvalue * unitfactor
				poolId.concInit = initvalue
			else:
				nr = model.getNumRules()
				found = False
				for nrItem in range(0,nr):
					rule = model.getRule(nrItem)
					assignRule = rule.isAssignment()
					if ( assignRule ):
						rule_variable = rule.getVariable()
						if (rule_variable == sId):
							found = True
							break
				if not (found):
					print "Invalid SBML: Either initialConcentration or initialAmount must be set or it should be found in assignmentRule but non happening for ",sName
					return False	
	return True

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

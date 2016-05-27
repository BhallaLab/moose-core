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
	filepath,filenameExt = os.path.split(filename)
	filename = filenameExt[:filenameExt.find('.')]
	
	#validatemodel
	sbmlOk = False
	global spe_constTrue
	spe_constTrue = []
	global nameList_
	nameList_ = []

	xmlns = XMLNamespaces()
	xmlns.add("http://www.sbml.org/sbml/level3/version1")
	xmlns.add("http://www.moose.ncbs.res.in","moose")
	xmlns.add("http://www.w3.org/1999/xhtml","xhtml")
	sbmlDoc.setNamespaces(xmlns)
	cremodel_ = sbmlDoc.createModel()
	cremodel_.setId(filename)
	cremodel_.setTimeUnits("second")
	cremodel_.setExtentUnits("substance")
	cremodel_.setSubstanceUnits("substance")
	
	writeUnits(cremodel_)
	modelAnno = writeSimulationAnnotation(modelpath)
	if modelAnno:
		cremodel_.setAnnotation(modelAnno)

	SBMLok  = validateModel( sbmlDoc )
	if ( SBMLok ):
		filepath = '/home/harsha/Trash/python'
		oo = writeSBMLToString(sbmlDoc)
		gg = filepath+"/"+filename+'.xml'

		writeSBMLToFile( sbmlDoc, gg)
		return gg
	if ( not SBMLok ):
		cerr << "Errors encountered " << endl;
		return -1;
	

def writeUnits(cremodel_):
	unitVol = cremodel_.createUnitDefinition()
	unitVol.setId( "volume")
	unit = unitVol.createUnit()
	unit.setKind(UNIT_KIND_LITRE)
	unit.setMultiplier(1.0)
	unit.setExponent(1.0)
	unit.setScale(0)

	unitSub = cremodel_.createUnitDefinition()
	unitSub.setId("substance")
	unit = unitSub.createUnit()
	unit.setKind( UNIT_KIND_ITEM )
	unit.setMultiplier(1)
	unit.setExponent(1.0)
	unit.setScale(0)
	
#write Simulation runtime,simdt,plotdt 
def writeSimulationAnnotation(modelpath):
	modelAnno = ""
	if moose.exists(modelpath+'/info'):
		mooseclock = moose.Clock('/clock')
		modelAnno ="<moose:ModelAnnotation>\n"
		modelAnnotation = moose.element(modelpath+'/info')
		modelAnno = modelAnno+"<moose:ModelTime> "+str(modelAnnotation.runtime)+" </moose:ModelTime>\n"
		modelAnno = modelAnno+"<moose:ModelSolver> "+modelAnnotation.solver+" </moose:ModelSolver>\n"
		modelAnno = modelAnno+"<moose:simdt>"+ str(mooseclock.dts[11]) + " </moose:simdt>\n";
		modelAnno = modelAnno+"<moose:plotdt> " + str(mooseclock.dts[18]) +" </moose:plotdt>\n";
		plots = "";
		graphs = moose.wildcardFind(modelpath+"/##[TYPE=Table2]")
		for gphs in range(0,len(graphs)):
			gpath = graphs[gphs].neighbors['requestOut']
			if len(gpath) != 0:
				q = moose.element(gpath[0])
				ori = q.path
				graphSpefound = False
				while not(isinstance(moose.element(q),moose.CubeMesh)):
					q = q.parent
					graphSpefound = True
				if graphSpefound:
					if not plots:
						plots = ori[ori.find(q.name)-1:len(ori)]
					else:
						plots = plots + "; "+ori[ori.find(q.name)-1:len(ori)]
		if plots != " ":
			modelAnno = modelAnno+ "<moose:plots> "+ plots+ "</moose:plots>\n";
		modelAnno = modelAnno+"</moose:ModelAnnotation>"
	return modelAnno

def validateModel( sbmlDoc ):
	#print " sbmlDoc ",sbmlDoc.toSBML()
	if ( not sbmlDoc ):
		print "validateModel: given a null SBML Document"
		return False

	consistencyMessages    = ""
	validationMessages     = ""
	noProblems             = True
	numCheckFailures       = 0
	numConsistencyErrors   = 0
	numConsistencyWarnings = 0
	numValidationErrors    = 0
	numValidationWarnings  = 0
	#Once the whole model is done and before it gets written out, 
	#it's important to check that the whole model is in fact complete, consistent and valid.
	numCheckFailures = sbmlDoc.checkInternalConsistency()
	if ( numCheckFailures > 0 ):
		noProblems = False
		for i in range(0,numCheckFailures ):
			sbmlErr = sbmlDoc.getError(i);
			if ( sbmlErr.isFatal() or sbmlErr.isError() ):
				++numConsistencyErrors;
			else:
				++numConsistencyWarnings

		constStr = sbmlDoc.printErrors()
		consistencyMessages = constStr
	  
	#If the internal checks fail, it makes little sense to attempt
	#further validation, because the model may be too compromised to
	#be properly interpreted.
	if ( numConsistencyErrors > 0 ):
		consistencyMessages += "Further validation aborted.";
	else:
		numCheckFailures = sbmlDoc.checkConsistency()
		#numCheckFailures = sbmlDoc.checkL3v1Compatibility() 
		if ( numCheckFailures > 0 ):
			noProblems = False;
			for i in range(0, (numCheckFailures ) ):
				sbmlErr = sbmlDoc.getError(i);
				print " Err ",sbmlErr
				if ( sbmlErr.isFatal() or sbmlErr.isError() ):
					++numValidationErrors;
				else:
					++numValidationWarnings;
		oss = sbmlDoc.printErrors()
		validationMessages = oss
	if ( noProblems ):
		return True
	else:
		if ( numConsistencyErrors > 0 ):
			if numConsistencyErrors == 1: t = "" 
			else: t="s"          
			print "ERROR: encountered " + numConsistencyErrors + " consistency error" +t+ " in model '" + sbmlDoc.getModel().getId() + "'."
	if ( numConsistencyWarnings > 0 ):
		if numConsistencyWarnings == 1:
			t1 = "" 
		else: t1 ="s"
		print "Notice: encountered " + numConsistencyWarnings +" consistency warning" + t + " in model '" + sbmlDoc.getModel().getId() + "'."
	  
	if ( numValidationErrors > 0 ):
		if numValidationErrors == 1:
			t2 = "" 
		else: t2 ="s" 
		print "ERROR: encountered " + numValidationErrors  + " validation error" + t2 + " in model '" + sbmlDoc.getModel().getId() + "'."
		
		if ( numValidationWarnings > 0 ):
			if numValidationWarnings == 1:
				t3 = "" 
			else: t3 = "s"

			print "Notice: encountered " + numValidationWarnings + " validation warning" + t3 + " in model '" + sbmlDoc.getModel().getId() + "'." 
		
		print validationMessages;
	return ( numConsistencyErrors == 0 and numValidationErrors == 0 )

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
	
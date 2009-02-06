/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "moose.h"
#include <sbml/SBMLTypes.h>
#include "element/Neutral.h"
#include "element/Wildcard.h"
#include "kinetics/KinCompt.h"
#include <sstream>
#include <set>
#include <algorithm>
#include "SbmlWriter.h"

void SbmlWriter::write(string filename,Id location)
{
  	SBMLDocument* sbmlDoc = 0;
  	bool SBMLok = false;
	//bool SBMLok = true;
	sbmlDoc = createModel(filename); 
  	SBMLok  = validateModel(sbmlDoc);
	filename += ".xml";
    	if (SBMLok) 
		writeModel(sbmlDoc, filename);
    	delete sbmlDoc;
	if (!SBMLok) {
		cerr << "Errors encountered " << endl;
		return ;
	}
}

/**
 * Create an SBML model in the SBML Level 2 Version 3 Specification.
 */

SBMLDocument* SbmlWriter::createModel(string filename)
{
	//SBMLDocument sbmlDoc = new SBMLDocument();
	//sbmlDoc.setLevelAndVersion( 2, 3 );	
	// Create an SBMLDocument object using in Level 2 Version 3 format:
	SBMLDocument* sbmlDoc = new SBMLDocument(2, 3);
	Model* model = sbmlDoc->createModel();
  	model->setId(filename);
  
  	UnitDefinition* unitdef;
 	Unit* unit;

  	// (UnitDefinition1) Create an UnitDefinition object for "per_second".

  	unitdef = model->createUnitDefinition();
  	unitdef->setId("per_second");

  	// Create an Unit inside the UnitDefinition object above:

  	unit = unitdef->createUnit();
  	unit->setKind(UNIT_KIND_SECOND);
  	unit->setExponent(-1);

  	// (UnitDefinition2) Create an UnitDefinition object for "litre_per_mole_per_second".  
    
  	unitdef = model->createUnitDefinition();
 	unitdef->setId("litre_per_mole_per_second");
    
	// Create individual unit objects that will be put inside
  	// the UnitDefinition to compose "litre_per_mole_per_second".

	 unit = unitdef->createUnit();
	 unit->setKind(UNIT_KIND_MOLE);
	 unit->setExponent(-1);

	 unit = unitdef->createUnit();
	 unit->setKind(UNIT_KIND_LITRE);
	 unit->setExponent(1);

	 unit = unitdef->createUnit();
	 unit->setKind(UNIT_KIND_SECOND);
	 unit->setExponent(-1);

 	// Create a string for the identifier of the compartment.  

	static const Cinfo* kincomptCinfo = initKinComptCinfo();
	static const Finfo* sizeFinfo = kincomptCinfo->findFinfo( "size" );
	static const Finfo* dimensionFinfo = kincomptCinfo->findFinfo( "numDimensions" );
	Eref comptEl;
	vector< Id > compts;
	vector< Id >::iterator itr;
	wildcardFind("/kinetics/##[TYPE=KinCompt]", compts);
 	vector< Eref > outcompt;	
	for (itr = compts.begin(); itr != compts.end(); itr++)
	{
		comptEl = ( *itr )();
		::Compartment* compt = model->createCompartment();
		const string comptName = comptEl.name();
		compt->setId(comptName);		
		double size;
               	get<double>(comptEl, sizeFinfo, size);
		unsigned int numD;
		get<unsigned int>(comptEl, dimensionFinfo, numD);
		compt->setSpatialDimensions(numD);
		if (numD == 3)		
			compt->setSize(size*1e3);
		else
			compt->setSize(size);
		
		outcompt.clear();
		targets(comptEl,"outside",outcompt);
		if ( outcompt.size() > 1 )
			cerr << "Warning: SbmlWriter: Too many outer compartments for " << comptName << ". Ignoring all but one.\n";
		
		if ( outcompt.size() >= 1 ) {
			string outName = outcompt[ 0 ].name();
			compt->setOutside(outName);
			
		}
		// Create the Species objects inside the Model object. 
		static const Cinfo* moleculeCinfo = initMoleculeCinfo();	
		static const Finfo* nInitFinfo = moleculeCinfo->findFinfo( "nInit" );	
		static const Finfo* concInitFinfo = moleculeCinfo->findFinfo( "concInit" );
		static const Finfo* modeFinfo = moleculeCinfo->findFinfo( "mode" );
		Eref moleEl;	
		vector< Id > molecs;
		vector< Id >::iterator mitr;
		string comptPath = comptEl.id().path();
		wildcardFind(comptPath + "/#[TYPE=Molecule]", molecs);
		for (mitr = molecs.begin(); mitr != molecs.end(); mitr++)
		{		
			moleEl = ( *mitr )();		
			Species *sp = model->createSpecies();
			sp->setId((moleEl)->name());
			Id parent=Neutral::getParent(moleEl);
			string parentCompt=parent()->name();
			sp->setCompartment(parentCompt);
			int mode;		
			get< int >(moleEl,modeFinfo,mode);
			if (mode==0){
				sp->setConstant(false);
				sp->setBoundaryCondition(false); 
			}	
			else{
				sp->setConstant(true);
				sp->setBoundaryCondition(true); 
			}
			unsigned int dimension;
		        get< unsigned int >(parent.eref(),dimensionFinfo,dimension);
			double initamt=0.0;
			get< double >(moleEl,nInitFinfo,initamt); 
			//get < double > (moleEl,concInitFinfo,initconc);
			//double ninit=initconc/6e23 * size/1e-3;
			initamt /= 6e23 ;
			sp->setInitialAmount(initamt);
			/*
			bool has_subunits = true;
			if (dimension > 0 || has_subunits == false)
				sp->setInitialConcentration(initconc);
			else	
				sp->setInitialAmount(initamt);*/
		}
  		static const Cinfo* reactionCinfo = initReactionCinfo();
 		static const Finfo* kbFinfo = reactionCinfo->findFinfo( "kb" );	
		static const Finfo* kfFinfo = reactionCinfo->findFinfo( "kf" );	
	  	Eref rectnEl;
		vector< Id > reaction;
		vector< Id >::iterator ritr;
		vector< Eref > rct;
		vector< Eref > pdt;
		wildcardFind(comptPath + "/#[TYPE=Reaction]", reaction);
		for (ritr = reaction.begin(); ritr != reaction.end(); ritr++)
		{
			rectnEl = ( *ritr )();		
			Reaction* reaction;
	  		SpeciesReference* spr;
	 		KineticLaw* kl;
			Parameter*  para; 
	  		
	  		// Create Reactant objects inside the Reaction object 

	 		reaction = model->createReaction();
			reaction->setId((rectnEl)->name());
			cout<<"reaction :"<<(rectnEl)->name()<<endl;
			
			double kb=0.0,kf=0.0;
			get< double >( rectnEl, kbFinfo, kb); 
			get< double >( rectnEl, kfFinfo, kf); 
			if (kb == 0.0)
	 			reaction->setReversible(false);
			else
				reaction->setReversible(true);

			//  Create a Reactant object that references particular Species in the model.  
		
			rct.clear();
			targets(rectnEl,"sub",rct);
			std::set< Eref > rctUniq ;
			double rctstoch ;
			rctUniq.insert(rct.begin(),rct.end());
			std::set< Eref >::iterator ri;
			double rct_order = 0.0;
			/*for (int i=0; i<rct.size();i++)
			{	
				spr = reaction->createReactant();
				string rctName = rct[ i ].name();		
				spr->setSpecies(rctName);
				
			}*/
			double frate =1.0,brate=1.0;
			for(ri=rctUniq.begin(); ri != rctUniq.end(); ri++)
			{	
				spr = reaction->createReactant();
				//rct_order = 0.0;
				string rctName = (*ri).name();	
				//cout<<"rct name :"<<rctName<<endl;	
				spr->setSpecies(rctName);				
				rctstoch=count(rct.begin(),rct.end(),*ri);
				spr->setStoichiometry(rctstoch);
				//cout<<"stoichiometry :"<<rctstoch<<endl;
				rct_order += rctstoch;
				double initconc;
				get < double > ((*ri),concInitFinfo,initconc);

				cout<<"conc is "<<initconc<<"exponent is "<<rctstoch<<endl;
				frate *= pow(initconc,rctstoch); 
				
			}
			//cout<<"rct_order is "<<rct_order<<endl;
			// Create a Product object that references particular Species  in the model. 		
			pdt.clear();
			targets(rectnEl,"prd",pdt);
	  		/*for (int i=0; i<pdt.size();i++)
			{	
				spr = reaction->createProduct();
				string pdtName = pdt[ i ].name();
	 		 	spr->setSpecies(pdtName);
				
			}*/
			
			std::set < Eref > pdtUniq;
			double pdtstoch;
			pdtUniq.insert(pdt.begin(),pdt.end());
			std::set < Eref > ::iterator pi;
			double pdt_order = 0.0;
			for(pi = pdtUniq.begin(); pi != pdtUniq.end(); pi++)
			{
				spr = reaction->createProduct();
				pdt_order = 0;
				string pdtName = (*pi).name();	
				//cout<<"pdt name :"<<pdtName<<endl;	
				spr->setSpecies(pdtName);				
				pdtstoch=count(pdt.begin(),pdt.end(),*pi);
				spr->setStoichiometry(pdtstoch);
				//cout<<"stoichiometry :"<<pdtstoch<<endl;
				pdt_order += pdtstoch;
				double initconc;
				get < double > ((*pi),concInitFinfo,initconc);
				cout<<"pdt exponent is "<<pdtstoch<<endl;
				brate *= pow(initconc,pdtstoch); 
			}
			//cout<<"pdt_order is "<<pdt_order<<endl;
			// Create a KineticLaw object inside the Reaction object 
			
			/*std::set< Eref > rctUniq ;
			int rctstoch ;
			rctUniq.insert(rct.begin(),rct.end());
			std::set< Eref >::iterator ri;*/
			ostringstream rate_law;
			if (kf != 0.0 ){
				//double NA = 6.02214199e23; //Avogardo's number		
				//double parmvalue = Kf*pow(6e23*size,rct_order-1)/(NA*size);
				//double parmvalue =kf/size;
				
				//cout<<"value "<<parmvalue<<endl;
				//rate_law <<comptName<<"*"<< "kf";
				rate_law <<comptName<<"*"<< "k";
				for(ri=rctUniq.begin(); ri != rctUniq.end(); ri++)
				{
					rctstoch=count(rct.begin(),rct.end(),*ri);
					rct_order += rctstoch;
					if (rctstoch == 1)
						rate_law <<"*"<< (*ri).name();
					else
						rate_law <<"*"<<(*ri).name()<<"^"<<rctstoch;
					
		
				} 
				
			}
			if (kb != 0.0){
				//double pvalue = Kb*pow(6e23*size,pdt_order-1)/(NA*size);
				//cout<<"pvalue is :"<<pvalue<<endl;
				//rate_law <<"-kb";				
				rate_law <<"-"<<"k";
				/*std::set < Eref > pdtUniq;
				int pdtstoch;
				pdtUniq.insert(pdt.begin(),pdt.end());
				std::set < Eref > ::iterator pi;*/
				for(pi = pdtUniq.begin(); pi != pdtUniq.end(); pi++)
				{
					pdtstoch=count(pdt.begin(),pdt.end(),*pi);
					if (pdtstoch == 1)
						rate_law <<"*"<< (*pi).name();
					else
						rate_law <<"*"<<(*pi).name()<<"^"<<pdtstoch;
					
		
				} 
			}
			cout<<"rate_law "<<rate_law.str()<<endl; 
			
			kl  = reaction->createKineticLaw();
			kl->setFormula(rate_law.str());

			/*string str = comptPath;
		       	string cPath = str.substr (10,str.length()); 
			ostringstream kparam;
			kparam<<cPath<<"_"<<(rectnEl)->name()<<"_"<<"k";
			cout<<"k string"<<kparam.str()<<endl; */

			// Create local Parameter objects inside the KineticLaw object. 
		
			para = kl->createParameter();
			double kval = kf/(size * frate );
			cout<<"kval "<<kval<<endl;
			para->setId("k");
			para->setValue(kval);
			//para->setUnits("litre_per_mole_per_second"); 
			
			
		}
		
 }
 return sbmlDoc;

}		

/**
 *
 * Writes the given SBMLDocument to the given file.
 *
 */ 
bool SbmlWriter::writeModel(const SBMLDocument* sbmlDoc, const string& filename)
{
	  SBMLWriter sbmlWriter;
	  bool result = sbmlWriter.writeSBML(sbmlDoc, filename);
	  if (result)
	  {
	  	cout << "Wrote file \"" << filename << "\"" << endl;
	  	return true;
	  }
	  else
	  {
	  	cerr << "Failed to write \"" << filename << "\"" << endl;
	  	return false;
  	  }
}

int SbmlWriter::targets(Eref object,const string& msg,vector< Eref >& target,const string& type )
{
	unsigned int oldSize = target.size();
	Eref found;
	Conn* i = object->targets( msg, 0 );
	for ( ; i->good(); i->increment() ) {
		found = i->target();
		if ( type != "" && !isType( found, type ) )	// speed this up
			continue;
		
		target.push_back( found );		
	}
	delete i;
	
	return target.size() - oldSize;
}

bool SbmlWriter::isType( Eref object, const string& type )
{
	return object->cinfo()->isA( Cinfo::find( type ) );
}

bool SbmlWriter::validateModel(SBMLDocument* sbmlDoc)
{
	  if (!sbmlDoc)
	  {
	    cerr << "validateModel: given a null SBML Document" << endl;
	    return false;
	  }
	 
	  string consistencyMessages;
	  string validationMessages;
	  bool noProblems                     = true;
	  unsigned int numCheckFailures       = 0;
	  unsigned int numConsistencyErrors   = 0;
	  unsigned int numConsistencyWarnings = 0;
	  unsigned int numValidationErrors    = 0;
	  unsigned int numValidationWarnings  = 0;

	  // Once the whole model is done and before it gets written out, 
	  // it's important to check that the whole model is in fact complete, consistent and valid.

	  numCheckFailures = sbmlDoc->checkInternalConsistency();
	  if ( numCheckFailures > 0 )
	  {
		    noProblems = false;
		    for (unsigned int i = 0; i < numCheckFailures; i++)
		    {
		      const SBMLError* sbmlErr = sbmlDoc->getError(i);
		      if ( sbmlErr->isFatal() || sbmlErr->isError() )
		      {
			++numConsistencyErrors;
		      }
		      else
		      {
			++numConsistencyWarnings;
		      }      
		    } 
		    ostringstream oss;
		    sbmlDoc->printErrors(oss);
		    consistencyMessages = oss.str(); 
	  }

	  // If the internal checks fail, it makes little sense to attempt
	  // further validation, because the model may be too compromised to
	  // be properly interpreted.

	  if (numConsistencyErrors > 0)
	  {
	    consistencyMessages += "Further validation aborted."; 
	  }
	  else
	  {
		    numCheckFailures = sbmlDoc->checkConsistency();
		    if ( numCheckFailures > 0 )
		    {
			      noProblems = false;
			      for (unsigned int i = 0; i < numCheckFailures; i++)
			      {
					const SBMLError* sbmlErr = sbmlDoc->getError(i);
					if ( sbmlErr->isFatal() || sbmlErr->isError() )
					{
					  ++numValidationErrors;
					}
					else
					{
					  ++numValidationWarnings;
					}      
			      } 
			      ostringstream oss;
			      sbmlDoc->printErrors(oss);
			      validationMessages = oss.str(); 
		    }
	  }

	  if (noProblems)
	    return true;
	  else
	  {
		    if (numConsistencyErrors > 0)
		    {
		      cout << "ERROR: encountered " << numConsistencyErrors 
			   << " consistency error" << (numConsistencyErrors == 1 ? "" : "s")
			   << " in model '" << sbmlDoc->getModel()->getId() << "'." << endl;
		    }
		    if (numConsistencyWarnings > 0)
		    {
		      cout << "Notice: encountered " << numConsistencyWarnings
			   << " consistency warning" << (numConsistencyWarnings == 1 ? "" : "s")
			   << " in model '" << sbmlDoc->getModel()->getId() << "'." << endl;
		    }
		    cout << endl << consistencyMessages;

		    if (numValidationErrors > 0)
		    {
		      cout << "ERROR: encountered " << numValidationErrors
			   << " validation error" << (numValidationErrors == 1 ? "" : "s")
			   << " in model '" << sbmlDoc->getModel()->getId() << "'." << endl;
		    }
		    if (numValidationWarnings > 0)
		    {
		      cout << "Notice: encountered " << numValidationWarnings
			   << " validation warning" << (numValidationWarnings == 1 ? "" : "s")
			   << " in model '" << sbmlDoc->getModel()->getId() << "'." << endl;
		    }
		    cout << endl << validationMessages;

		    return (numConsistencyErrors == 0 && numValidationErrors == 0);
	  }
}



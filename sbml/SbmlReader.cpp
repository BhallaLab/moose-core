/*******************************************************************
 * File:            SbmlReader.cpp
 * Description:      
 * Author:          
 * E-mail:          
 ********************************************************************/
/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include <cmath>
#include <sbml/SBMLTypes.h>
#include <sbml/UnitDefinition.h>
#include <sbml/units/UnitFormulaFormatter.h>
#include <sbml/units/FormulaUnitsData.h>
#include <string>

#include "SbmlReader.h"
#include <stdlib.h>

using namespace std;

/* read a model into MOOSE  */
int SbmlReader::read( string filename,string location )
{
#ifdef USE_SBML

  FILE * fp = fopen( filename.c_str(), "r" );
  if ( fp == NULL){
    cout << filename << " : File does not exist." << endl;
  }
  document_ = readSBML( filename.c_str() );
  unsigned num_errors = document_->getNumErrors();
  
  if ( num_errors > 0 )
    {
      cerr << "Errors encountered while reading" << endl;
      document_->printErrors( cerr );
      errorFlag_ = true;
      return;
    }
  model_= document_->getModel();
  cout << "level and version " << model_->getLevel() << model_->getVersion() << endl;
  if ( model_ == 0 )
    {
      cout << "No model present." << endl;
      errorFlag_ = true;
    }
	
  if ( !errorFlag_ )
    { 
      map< string,string > idMap;
      map< string,string > molMap;
    
      if ( !errorFlag_ )
	idMap = createCompartment( location );
      if ( !errorFlag_ )
	molMap = createMolecule( idMap );
      if ( !errorFlag_ )
	getRules();
      if ( !errorFlag_ )
	createReaction( molMap );
      /*if ( errorFlag_ )
		return;
#else
	cout << "This version does not have SBML support." << endl; 
	*/
    }
#endif
}
#ifdef USE_SBML

/* Pulling COMPARTMENT  */

map< string,string > SbmlReader::createCompartment( string location )
{

  map< string,string > idMap;	
  map< string,string > outsideMap;
  map< string,string > ::iterator iter;
  double msize = 0.0,size=0.0;	
  ::Compartment* compt;
  unsigned int num_compts = model_->getNumCompartments();
  cout << "num of compartments :" << num_compts <<endl;
  for ( unsigned int i = 0; i < num_compts; i++ )
    {
      compt = model_->getCompartment(i);
      std::string id = "";
      if ( compt->isSetId() ){
	id = compt->getId();
      }
      std::string name = "";
      if ( compt->isSetName() ){
	name = compt->getName();
      } 
      std::string outside = "";
      if ( compt->isSetOutside() ){
	outside = compt->getOutside ();
	
      }
      if ( compt->isSetSize() ){
	msize = compt->getSize();
      }
      cout << "Compartment " << id << " name: "<< name << " o: " << outside << " s: " <<msize << endl;
    }
    return idMap;
}

/* create MOLECULE */

map< string,string > SbmlReader::createMolecule( map< string,string> &idMap )
{	cout << "create Molecule" <<endl;
	map<string,string>molMap;
	int num_species = model_->getNumSpecies();
	cout << "num species: " << num_species << endl;
	for ( int sindex = 0; sindex < num_species; sindex++ )
	{
		Species* s = model_->getSpecies(sindex);
		if (!s){
			//cout << "species " << sindex << " is nul" << endl;
			continue;
		}
		std::string compt = "";		
		if ( s->isSetCompartment() )		
			compt = s->getCompartment();
		if (compt.length()< 1){
			cout << "compt is empty for species "<< sindex << endl;
			continue;
		}
		string id = s->getId();
		//cout<<"species is :"<<id<<endl;
		if (id.length() < 1){
			continue;
		}
		std::string name = "";
		if ( s->isSetName() ){
			name = s->getName();
		} 
		double initvalue =0.0;
		if ( s->isSetInitialConcentration() )
		  initvalue = s->getInitialConcentration();
		else if ( s->isSetInitialAmount() )
		  initvalue = s->getInitialAmount() ;
		else {
		  cout << "Invalid SBML: Either initialConcentration or initialAmount must be set." << endl;
		  return molMap;
		}
		unsigned int dimension;
		bool initconc = s->isSetInitialConcentration();
		bool cons=s->getConstant(); 
		bool bcondition = s->getBoundaryCondition();
		cout << " C: " << compt << " Species: " << id << " name "<< name <<  " inV " << initvalue << " conc " << initconc << "cons "  << cons << " bcon: " << bcondition << endl;
	}
	return molMap;

}
/*
*  Assignment Rule
*/

void SbmlReader::getRules()
{
	unsigned int nr = model_->getNumRules();
	cout<<"no of rules:"<<nr<<endl;
	for ( unsigned int r = 0;r < nr;r++ )
	{
		Rule * rule = model_->getRule(r);
		cout << "rule :" << rule->getFormula() << endl;
		bool assignRule = rule->isAssignment();
		cout << "is assignment :" << assignRule << endl;
		if ( assignRule ){
			string rule_variable = rule->getVariable();
			cout << "variable :" << rule_variable << endl;
		}
		bool rateRule = rule->isRate();
		if ( rateRule ){
			cout << "warning : for now Rate Rule is not handled " << endl;
			errorFlag_ = true;
		}
		bool  algebRule = rule->isAlgebraic(); 
		if ( algebRule ){
			cout << "warning: for now Algebraic Rule is not handled" << endl;
			errorFlag_ = true;
		}
	}
}

//REACTION
void SbmlReader::createReaction( map< string, string > &molMap )
{
  Reaction* reac;	
  for ( unsigned int r = 0; r < model_->getNumReactions(); r++ )
    {	
      reac = model_->getReaction( r ); 
      const string id=reac->getId();
      cout<<"reaction is "<<id<<endl;
      std::string name;
      if ( reac->isSetName() ){
	name = reac->getName();
      }

      if (reac->getNumModifiers() > 0)
	cout << "This is and enzymatics reaction need to deal with it " <<endl;
      else 
	{
	  bool rev=reac->getReversible();
	  bool fast=reac->getFast();
	  if ( fast ){
	    cout<<"warning: for now fast attribute is not handled"<<endl;
	    errorFlag_ = true;
	  }
	  int numRcts = reac->getNumReactants();
	  int numPdts = reac->getNumProducts();
	  double rctcount=0.0;	
	}//else
    }//reaction 
}//create reaction


#endif // USE_SBML

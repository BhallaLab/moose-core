/*******************************************************************
 * File:            SbmlWriter.h
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

#ifndef _SBMLWRITER_H
#define _SBMLWRITER_H
#include <sbml/SBMLTypes.h>

class SbmlWriter
{
		
	public:
		SbmlWriter() {;}
		~SbmlWriter() {;}
		int write( string filename, string location );
#ifdef USE_SBML
		void createModel( string filename, SBMLDocument& doc ,string target);
		bool validateModel(SBMLDocument* sbmlDoc );
		bool writeModel( const SBMLDocument* sbmlDoc, const string& filename );
		 
	private:
		Model* cremodel_;	
		string nameString( string str );
		string nameString1( string str );
		string changeName( string parent,string child );
		string idBeginWith(string name );
		string cleanNameId( Id itrid,int index);
		//string cleanNameId(id index);
		//~static int targets( Eref object, const string& msg,vector< Eref >& target,const string& type = "" );		
		//~ static bool isType( Eref object, const string& type );
		string parmUnit( double rct_order );
		//void getSubPrd(SpeciesReference* spr,vector < Id> subprdId, int index,ostringstream& rate_law,double &rct_order,bool w);
		void getSubPrd(Reaction* rec,string type,string enztype,Id itrRE, int index,ostringstream& rate_law,double &rct_order,bool w);
		void getModifier(ModifierSpeciesReference* mspr,vector < Id> mod, int index,ostringstream& rate_law,double &rct_order,bool w);
		//~ double transformUnits( double mvalue,UnitDefinition * ud );
		//~ string nameString( string str );
		//~ string changeName( string parent, string child );
		//~ string idBeginWith( string name );
		//~ string getParentFunc( Eref p );
		void printParameters( KineticLaw* kl,string k,double kvalue,string unit );
		string findNotes(Id itr);
		//~ void printReactants( Reaction* reaction,vector< Eref > sub,ostringstream& rlaw );
		//~ void printProducts( Reaction* reaction,vector< Eref > cplx,ostringstream& rlaw );
		//~ void printenzReactants( Reaction* reaction,vector< Eref > sub,ostringstream& rlaw,string parentCompt );
		//~ void printenzProducts( Reaction* reaction,vector< Eref > cplx,ostringstream& rlaw,string parentCompt );
		//~ void printEnzymes( vector< Id > enzms );
		//~ void getEnzyme( vector< Eref > enz,vector <string> &enzsName );
		//~ void getSubstrate( vector< Eref > sub,vector <string> &subsName );
		//~ void getProduct( vector< Eref > prd,vector <string> &prdsName );
#endif
};
//extern const Cinfo* initCinfo();
#endif // _SBMLWRITER_H

/*******************************************************************
 * File:            SbmlWriter.h
 * Description:      
 * Author:          Siji P George
 * E-mail:          siji.suresh@gmail.com
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
		void write( string filename,Id location );
#ifdef USE_SBML
		SBMLDocument* createModel( string filename );
		bool validateModel( SBMLDocument* sbmlDoc );
		bool writeModel( const SBMLDocument* sbmlDoc, const string& filename );

	private:
		Model* model_;	
		static int targets( Eref object, const string& msg,vector< Eref >& target,const string& type = "" );
		static bool isType( Eref object, const string& type );
		string parmUnit( double rct_order );
		double transformUnits( double mvalue,UnitDefinition * ud );
		string nameString( string str );
		string changeName( string parent, string child );
		string idBeginWith( string name );
		string getParentFunc( Eref p );
		void printParameters( KineticLaw* kl,string k,double kvalue,string unit );
		void printReactants( Reaction* reaction,vector< Eref > sub,ostringstream& rlaw );
		void printProducts( Reaction* reaction,vector< Eref > cplx,ostringstream& rlaw );
		void printenzReactants( Reaction* reaction,vector< Eref > sub,ostringstream& rlaw,string parentCompt );
		void printenzProducts( Reaction* reaction,vector< Eref > cplx,ostringstream& rlaw,string parentCompt );
		void printEnzymes( vector< Id > enzms );
		void getEnzyme( vector< Eref > enz,vector <string> &enzsName );
		void getSubstrate( vector< Eref > sub,vector <string> &subsName );
		void getProduct( vector< Eref > prd,vector <string> &prdsName );
#endif
};
extern const Cinfo* initKinComptCinfo();
extern const Cinfo* initMoleculeCinfo();
extern const Cinfo* initReactionCinfo();
extern const Cinfo* initEnzymeCinfo();
#endif // _SBMLWRITER_H

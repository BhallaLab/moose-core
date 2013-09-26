/*******************************************************************
 * File:            SbmlReader.h
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

#ifndef _SBMLREADER_H
#define _SBMLREADER_H
#ifdef USE_SBML

#include <sbml/SBMLTypes.h>

#include "../basecode/Id.h"
//class Id;
typedef struct
{
  Id enzyme;
  Id complex;
  vector<Id> substrates;
  vector<Id> products;
  double k1;
  double k2;
  double k3;
  int stage;
} EnzymeInfo;

class SbmlReader
{
	public:
		SbmlReader() {errorFlag_ = false;}
		~SbmlReader() {;}
		Id read(string filename,string location);
		map< string, Id> createCompartment(string location);
		map< string, Id> createMolecule(map<string,Id>&);
		void  createReaction(map<string,Id> &);
		
	private:
		bool errorFlag_;
 
		Model* model_;		
		SBMLDocument* document_;
		SBMLReader reader_;
		map< string, Id >elmtMolMap_;
		Id baseId;
		double transformUnits( double msize,UnitDefinition * ud,string type,bool hasonlySubUnit );
		void getRules();
		void printMembers( const ASTNode* p,vector <string> & ruleMembers );
		void addSubPrd(Reaction * reac,Id reaction_,string type);
		void getKLaw( KineticLaw * klaw,bool rev,vector< double > & rate );
		void pushParmstoVector( const ASTNode* p,vector <string> & parameters );
		void getParameters( const ASTNode* node,vector <string> & parameters );
		void setupMMEnzymeReaction( Reaction * reac,string id ,string name,map<string,Id> &);
		string getAnnotation( Reaction* reaction,map<string,EnzymeInfo> & );
		void setupEnzymaticReaction( const EnzymeInfo & einfo,string name,map< string, Id > & ,string name1);
		void findModelParent( Id cwe, const string& path,Id& parentId, string& modelName );
#endif
		
};

#endif // _SBMLREADER_H


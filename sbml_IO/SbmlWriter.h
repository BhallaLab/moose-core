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
		void write(string filename,Id location);
		SBMLDocument* createModel(string filename);
		bool validateModel(SBMLDocument* sbmlDoc);
		bool writeModel(const SBMLDocument* sbmlDoc, const string& filename);
	private:
		static int targets(Eref object,	const string& msg,vector< Eref >& target,const string& type = "" );
		static bool isType( Eref object, const string& type );
		string parmUnit(double rct_order);
		double transformUnits(double mvalue,UnitDefinition * ud);
		string nameString(string str);
		void printParameters(KineticLaw* kl,string k,double kvalue,string unit);
		void printReactants(Reaction* reaction,vector< Eref > enz,ostringstream& rlaw);
		void printProducts(Reaction* reaction,vector< Eref > cplx,ostringstream& rlaw);
		void printEnzymes(vector< Id > enzms,string parentCompt,double size,Model* model);
};
extern const Cinfo* initKinComptCinfo();
extern const Cinfo* initMoleculeCinfo();
extern const Cinfo* initReactionCinfo();
extern const Cinfo* initEnzymeCinfo();
#endif // _SBMLWRITER_H

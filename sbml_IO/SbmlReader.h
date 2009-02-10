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
#include <sbml/SBMLTypes.h>
class SbmlReader
{
	public:
		SbmlReader() {;}
		~SbmlReader() {;}
		void  read(string filename,Id location);
		void  createCompartment(Model* model,Id location);
		void  createMolecule(Model* model,map<string,Id> &);
		void  printParameter(Model* model);
		void  printUnit(Model* model);
		void  createReaction(Model* model,map<string,Id> &,map<string,Eref> &);
		
				
	private:
		SBMLDocument* document_;
		SBMLReader reader_;
		Element* comptEl_;
		Element* molecule_;
		Element* reaction_;
		string  prn_parm(const ASTNode* p);
		double transformUnits(double msize,UnitDefinition * ud);
		
		
};
extern const Cinfo* initKinComptCinfo();
extern const Cinfo* initMoleculeCinfo();
extern const Cinfo* initReactionCinfo();
#endif // _SBMLREADER_H


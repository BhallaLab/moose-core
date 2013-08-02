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
#include <sbml/SBMLTypes.h>


class SbmlReader
{
	struct EnzymeInfo;

	public:
		SbmlReader() {errorFlag_ = false;}
		~SbmlReader() {;}
		int read(string filename,string location);
#ifdef USE_SBML
		map< string, string > createCompartment(string location);
		map< string,string > createMolecule(map<string,string> &);
		void  createReaction(map<string,string> &);
		
#endif	// USE_SBML			
	private:
bool errorFlag_;
#ifdef USE_SBML
		Model* model_;		
		SBMLDocument* document_;
		SBMLReader reader_;
		void getRules();

#endif
		
};

#endif // _SBMLREADER_H


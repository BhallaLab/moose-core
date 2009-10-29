/*******************************************************************
 * File:            NeuromlWriter.h
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
#ifndef _NEUROMLWRITER_H
#define _NEUROMLWRITER_H
class NeuromlWriter
{
	public:
		NeuromlWriter() {;}
		~NeuromlWriter() {;}
		void writeModel( string filepath,Id location );
		//SBMLDocument* createModel( string filename );
		//bool validateModel( SBMLDocument* sbmlDoc );
		//bool writeModel( const SBMLDocument* sbmlDoc, const string& filename );



};
extern const Cinfo* initCompartmentCinfo();
extern const Cinfo* initHHChannelCinfo();
#endif // _NEUROMLWRITER_H


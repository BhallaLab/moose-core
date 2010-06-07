/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _CHEM_COMPT_H
#define _CHEM_COMPT_H

class ChemCompt: public Data
{
	public: 
		ChemCompt();
		//////////////////////////////////////////////////////////////////
		// Field assignment stuff
		//////////////////////////////////////////////////////////////////

		void setSize( double v );
		double getSize() const;

		void setDimensions( unsigned int v );
		unsigned int getDimensions() const;

		//////////////////////////////////////////////////////////////////
		// Dest funcs
		//////////////////////////////////////////////////////////////////

		void process( const ProcInfo* p, const Eref& e );

		static const Cinfo* initCinfo();
	private:
		double size_;
		unsigned int dimensions_;
};

#endif	// _CHEM_COMPT_H

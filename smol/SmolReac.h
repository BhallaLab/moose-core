/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _SMOL_REAC_H
#define _SMOL_REAC_H

class SmolReac: public SmolSim
{
	public: 
		SmolReac();
		~SmolReac();

		//////////////////////////////////////////////////////////////////
		// Field assignment stuff
		//////////////////////////////////////////////////////////////////

		void setKf( const Eref& e, const Qinfo* q, double v );
		double getKf( const Eref& e, const Qinfo* q ) const;
		void setKb( const Eref& e, const Qinfo* q, double v );
		double getKb( const Eref& e, const Qinfo* q ) const;

		//////////////////////////////////////////////////////////////////
		// Dest funcs
		//////////////////////////////////////////////////////////////////

		void process( const Eref& e, ProcPtr p );
		void reinit( const Eref& e, ProcPtr p );
		void sub( double v );
		void prd( double v );

		//////////////////////////////////////////////////////////////////
		// utility funcs
		//////////////////////////////////////////////////////////////////
		static void zombify( Element* solver, Element* orig );
		static void unzombify( Element* zombie );

		static const Cinfo* initCinfo();
	private:
};

#endif	// _SMOL_REAC_H

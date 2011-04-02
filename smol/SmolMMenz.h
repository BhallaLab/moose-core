/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _SMOL_MM_ENZ_H
#define _SMOL_MM_ENZ_H

/**
 * This class represents the Michaelis-Menten type enzyme, obeying the
 * equation
 * V = kcat.[Etot].[S]/( Km + [S] )
 */
class SmolMMenz: public SmolSim
{
	public: 
		SmolMMenz();

		//////////////////////////////////////////////////////////////////
		// Field assignment stuff
		//////////////////////////////////////////////////////////////////

		void setKm( const Eref& e, const Qinfo* q, double v );
		double getKm( const Eref& e, const Qinfo* q ) const;
		void setKcat( const Eref& e, const Qinfo* q, double v );
		double getKcat( const Eref& e, const Qinfo* q ) const;

		//////////////////////////////////////////////////////////////////
		// Dest funcs
		//////////////////////////////////////////////////////////////////

		void process( const Eref& e, ProcPtr p );
		void reinit( const Eref& e, ProcPtr p );
		void dummy( double n );

		//////////////////////////////////////////////////////////////////
		// Utility  funcs
		//////////////////////////////////////////////////////////////////

		static void zombify( Element* solver, Element* orig );
		static void unzombify( Element* zombie );

		static const Cinfo* initCinfo();
	private:
};

#endif // _SMOL_MM_ENZ_H

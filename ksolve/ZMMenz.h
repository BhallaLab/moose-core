/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _Z_MM_ENZ_H
#define _Z_MM_ENZ_H

/**
 * This class represents the Michaelis-Menten type enzyme, obeying the
 * equation
 * V = kcat.[Etot].[S]/( Km + [S] )
 */
class ZMMenz: public EnzBase
{
	public: 
		ZMMenz();

		//////////////////////////////////////////////////////////////////
		// Field assignment stuff: All override virtual funcs
		//////////////////////////////////////////////////////////////////

		void vSetKm( const Eref& e, const Qinfo* q, double v );
		double vGetKm( const Eref& e, const Qinfo* q ) const;
		void vSetNumKm( const Eref& e, const Qinfo* q, double v );
		double vGetNumKm( const Eref& e, const Qinfo* q ) const;
		void vSetKcat( const Eref& e, const Qinfo* q, double v );
		double vGetKcat( const Eref& e, const Qinfo* q ) const;

		//////////////////////////////////////////////////////////////////
		// Dest funcs: All override virtual funcs
		//////////////////////////////////////////////////////////////////

		void vRemesh( const Eref& e, const Qinfo* q );

		//////////////////////////////////////////////////////////////////
		// Utility  funcs
		//////////////////////////////////////////////////////////////////

		/// Does ZMMenz specific functions during conversion to zombie
		/// virtual func overrides default.
		void setSolver( Id solver, Id orig );

		static const Cinfo* initCinfo();
	private:
		SolverBase* solver_;
		double Km_; /// Km in conc units: millimolar
};

#endif // _Z_MM_ENZ_H

/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _MM_ENZ_H
#define _MM_ENZ_H

/**
 * This class represents the Michaelis-Menten type enzyme, obeying the
 * equation
 * V = kcat.[Etot].[S]/( Km + [S] )
 */
class MMenz: public EnzBase
{
	public:
		MMenz();
		virtual ~MMenz();

		//////////////////////////////////////////////////////////////////
		// Field assignment stuff
		//////////////////////////////////////////////////////////////////

		void vSetKm( const Eref& e, double v );
		void vSetKcat( const Eref& e, double v );

		//////////////////////////////////////////////////////////////////
		// Dest funcs
		//////////////////////////////////////////////////////////////////
		// sub, prd, enz, remesh are all defined in EnzBase as dummies.

		void setSolver( const Eref& e, ObjId solver );
		static const Cinfo* initCinfo();
	private:
		Stoich* stoich_;
		double Km_; /// Km in Concentration units, millimolar.
};

#endif // MM_ENZ_H

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
class MMenz
{
	public: 
		MMenz();

		//////////////////////////////////////////////////////////////////
		// Field assignment stuff
		//////////////////////////////////////////////////////////////////

		void setKm( const Eref& e, const Qinfo* q, double v );
		double getKm( const Eref& e, const Qinfo* q ) const;
		void setNumKm( const Eref& e, const Qinfo* q, double v );
		double getNumKm( const Eref& e, const Qinfo* q ) const;
		unsigned int getNumSub( const Eref& e, const Qinfo* q ) const;
		void setKcat( double v );
		double getKcat() const;

		//////////////////////////////////////////////////////////////////
		// Dest funcs
		//////////////////////////////////////////////////////////////////

		void process( const Eref& e, ProcPtr p );
		void reinit( const Eref& e, ProcPtr p );
		void sub( double n );
		void enz( double n );
		void prd( double n );

		static const Cinfo* initCinfo();
	private:
		double Km_; /// Km in Concentration units, millimolar.
		double numKm_; /// Km in number units
		double kcat_; /// kcat in 1/sec
		double sub_;	/// State variable: substrate (in num units) * numKm
		double enz_;	/// State variable: enz number.
};

#endif // MM_ENZ_H

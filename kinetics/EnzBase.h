/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _ENZ_BASE_H
#define _ENZ_BASE_H

/**
 * This class is the base class for enzymes, both the classical Michaelis-
 * Menten form and the form with explicit enz-substrate complexes that
 * MOOSE prefers.
 */
class EnzBase
{
	public:
		EnzBase();
		virtual ~EnzBase();

		//////////////////////////////////////////////////////////////////
		// Field assignment stuff
		//////////////////////////////////////////////////////////////////
		void setKm( const Eref& e, double v );
		double getKm( const Eref& e ) const;
		void setNumKm( const Eref& e, double v );
		double getNumKm( const Eref& e ) const;
		void setKcat( const Eref& e, double v );
		double getKcat( const Eref& e ) const;

		// This doesn't need a virtual form, depends on standard msgs.
		unsigned int getNumSub( const Eref& e ) const;
		unsigned int getNumPrd( const Eref& e ) const;

		//////////////////////////////////////////////////////////////////
		// Virtual funcs for field assignment stuff
		//////////////////////////////////////////////////////////////////
		virtual void vSetKm( const Eref& e, double v ) = 0;
		virtual void vSetKcat( const Eref& e, double v ) = 0;

		//////////////////////////////////////////////////////////////////
		// Dest funcs
		//////////////////////////////////////////////////////////////////
		void sub( double n );	// Now these are all empty functions.
		void enz( double n );
		void prd( double n );
		
		//////////////////////////////////////////////////////////////////

		static const Cinfo* initCinfo();
	protected:
		double Km_;
		double kcat_;
};

#endif // ENZ_BASE_H

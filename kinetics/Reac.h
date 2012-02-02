/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _REAC_H
#define _REAC_H

class Reac
{
	public: 
		Reac();
		Reac( double kf, double kb );

		//////////////////////////////////////////////////////////////////
		// Field assignment stuff
		//////////////////////////////////////////////////////////////////

		void setNumKf( const Eref&e, const Qinfo* q, double v );
		double getNumKf( const Eref& e, const Qinfo *q ) const;
		void setNumKb( const Eref&e, const Qinfo* q, double v );
		double getNumKb( const Eref& e, const Qinfo *q ) const;

		/// set Kf in concentration units
		void setConcKf( const Eref& e, const Qinfo* q, double v );
		/// get Kf in concentration units
		double getConcKf( const Eref& e, const Qinfo* q ) const;

		/// set Kb in concentration units
		void setConcKb( const Eref& e, const Qinfo* q, double v );
		/// get Kb in concentration units
		double getConcKb( const Eref& e, const Qinfo* q ) const;

		/// Get number of substrates
		unsigned int getNumSub( const Eref& e, const Qinfo* q ) const;
		/// Get number of products
		unsigned int getNumPrd( const Eref& e, const Qinfo* q ) const;

		//////////////////////////////////////////////////////////////////
		// Dest funcs
		//////////////////////////////////////////////////////////////////

		void process( const Eref& e, ProcPtr p );
		void reinit( const Eref& e, ProcPtr p );
		void sub( double v );
		void prd( double v );
		void remesh( const Eref& e, const Qinfo* q );

		static const Cinfo* initCinfo();
	private:
		double kf_;	// Used for EE method, but secondary to the ConcKf
		double kb_;	// Used for EE method, but secondary to the ConcKf
		double concKf_;	// Kf in concentration and time units
		double concKb_;	// Kb in concentration and time units
		double sub_;	// State variable
		double prd_;	// State variable
};

#endif // REAC_H

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

		void setKf( double v );
		double getKf() const;
		void setKb( double v );
		double getKb() const;

		/// set Kf in concentration units
		void setConcKf( const Eref& e, const Qinfo* q, double v );
		/// get Kf in concentration units
		double getConcKf( const Eref& e, const Qinfo* q ) const;

		/// set Kb in concentration units
		void setConcKb( const Eref& e, const Qinfo* q, double v );
		/// get Kb in concentration units
		double getConcKb( const Eref& e, const Qinfo* q ) const;

		//////////////////////////////////////////////////////////////////
		// Dest funcs
		//////////////////////////////////////////////////////////////////

		void process( const Eref& e, ProcPtr p );
		void reinit( const Eref& e, ProcPtr p );
		void sub( double v );
		void prd( double v );

		static const Cinfo* initCinfo();
	private:
		double kf_;
		double kb_;
		double sub_;
		double prd_;
};

#endif // REAC_H

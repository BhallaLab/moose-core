/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

class Reac: public Data
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

		//////////////////////////////////////////////////////////////////
		// Dest funcs
		//////////////////////////////////////////////////////////////////

		void process( const ProcInfo* p, const Eref& e );
		void eprocess( Eref e, const Qinfo* q, ProcInfo* p );
		void reinit( const Eref& e, const Qinfo*q, ProcInfo* p );
		void sub( double v );
		void prd( double v );

		static const Cinfo* initCinfo();
	private:
		double kf_;
		double kb_;
		double sub_;
		double prd_;
};

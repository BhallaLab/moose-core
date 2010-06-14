/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _ENZ_H
#define _ENZ_H

class Enz: public Data
{
	public: 
		Enz();

		//////////////////////////////////////////////////////////////////
		// Field assignment stuff
		//////////////////////////////////////////////////////////////////

		void setK1( double v );
		double getK1() const;
		void setK2( double v );
		double getK2() const;
		void setK3( double v );
		double getK3() const;

		//////////////////////////////////////////////////////////////////
		// Dest funcs
		//////////////////////////////////////////////////////////////////

		void process( const ProcInfo* p, const Eref& e );
		void eprocess( Eref e, const Qinfo* q, ProcInfo* p );
		void reinit( const Eref& e, const Qinfo*q, ProcInfo* p );
		void sub( double n );
		void enz( double n );
		void prd( double n );
		void cplx( double n );

		static const Cinfo* initCinfo();
	private:
		double k1_;
		double k2_;
		double k3_;
		double sub_;
		double prd_;
		double cplx_;
};

#endif // ENZ_H

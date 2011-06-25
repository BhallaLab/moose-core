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

class Enz
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

		void setKm( const Eref& e, const Qinfo* q, double v );
		double getKm( const Eref& e, const Qinfo* q ) const;
		void setRatio( double v );
		double getRatio() const;

		//////////////////////////////////////////////////////////////////
		// Dest funcs
		//////////////////////////////////////////////////////////////////

		void process( const Eref& e, ProcPtr p );
		void reinit( const Eref& e, ProcPtr p );
		void sub( double n );
		void enz( double n );
		void prd( double n );
		void cplx( double n );

		static const Cinfo* initCinfo();
	private:
		double k1_;
		double k2_;
		double k3_;
		double r1_;
		double r2_;
		double r3_;
};

#endif // ENZ_H

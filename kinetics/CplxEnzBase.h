/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _CPLX_ENZ_BASE_H
#define _CPLX_ENZ_BASE_H

class CplxEnzBase: public EnzBase
{
	public: 
		CplxEnzBase();
		virtual ~CplxEnzBase();

		//////////////////////////////////////////////////////////////////
		// Field assignment stuff
		//////////////////////////////////////////////////////////////////
		void setK1( const Eref& e, const Qinfo* q, double v );
		double getK1( const Eref& e, const Qinfo* q ) const;
		void setK2( const Eref& e, const Qinfo* q, double v );
		double getK2( const Eref& e, const Qinfo* q ) const;
		//void setK3( const Eref& e, const Qinfo* q, double v );
		//double getK3( const Eref& e, const Qinfo* q ) const;
		void setRatio( const Eref& e, const Qinfo* q, double v );
		double getRatio( const Eref& e, const Qinfo* q ) const;
		void setConcK1( const Eref& e, const Qinfo* q, double v );
		double getConcK1( const Eref& e, const Qinfo* q ) const;

		//////////////////////////////////////////////////////////////////
		// Virtual field stuff to use as base class for Enz and ZombieEnz.
		//////////////////////////////////////////////////////////////////
		virtual void vSetK1( const Eref& e, const Qinfo* q, double v ) = 0;
		virtual double vGetK1( const Eref& e, const Qinfo* q ) const = 0;
		virtual void vSetK2( const Eref& e, const Qinfo* q, double v ) = 0;
		virtual double vGetK2( const Eref& e, const Qinfo* q ) const = 0;
		virtual void vSetRatio( const Eref& e, const Qinfo* q, double v ) = 0;
		virtual double vGetRatio( const Eref& e, const Qinfo* q ) const = 0;
		virtual void vSetConcK1( const Eref& e, const Qinfo* q, double v ) = 0;
		virtual double vGetConcK1( const Eref& e, const Qinfo* q ) const = 0;
		//////////////////////////////////////////////////////////////////
		// A new Dest function.
		//////////////////////////////////////////////////////////////////
		void cplx( double n );
		//////////////////////////////////////////////////////////////////
		// The other dest funcs, all virtual, come from EnzBase.
		//////////////////////////////////////////////////////////////////
		virtual void vCplx( double n );

		//////////////////////////////////////////////////////////////////
		// Zombification stuff, overrides the one from EnzBase.
		//////////////////////////////////////////////////////////////////
		static void zombify( Element* original, const Cinfo* zClass,
			Id solver );

		static const Cinfo* initCinfo();
	private:
};

#endif // CPLX_ENZ_BASE_H

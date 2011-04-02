/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _SMOL_ENZ_H
#define _SMOL_ENZ_H

class SmolEnz: public SmolSim
{
	public: 
		SmolEnz();

		//////////////////////////////////////////////////////////////////
		// Field assignment stuff
		//////////////////////////////////////////////////////////////////

		void setK1( const Eref& e, const Qinfo* q, double v );
		double getK1( const Eref& e, const Qinfo* q ) const;
		void setK2( const Eref& e, const Qinfo* q, double v );
		double getK2( const Eref& e, const Qinfo* q ) const;
		void setK3( const Eref& e, const Qinfo* q, double v );
		double getK3( const Eref& e, const Qinfo* q ) const;

		//////////////////////////////////////////////////////////////////
		// Dest funcs
		//////////////////////////////////////////////////////////////////

		void process( const Eref& e, ProcPtr p );
		void reinit( const Eref& e, ProcPtr p );
		void dummy( double n );

		//////////////////////////////////////////////////////////////////
		// Utility  funcs
		//////////////////////////////////////////////////////////////////
		static void zombify( Element* solver, Element* orig );
		static void unzombify( Element* zombie );

		static const Cinfo* initCinfo();
	private:
};

#endif // _SMOL_ENZ_H

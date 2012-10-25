/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _Z_POOL_H
#define _Z_POOL_H

class ZPool: public PoolBase
{
	public: 
		ZPool();
		~ZPool();

		//////////////////////////////////////////////////////////////////
		// Field assignment stuff
		//////////////////////////////////////////////////////////////////

		void vSetN( const Eref& e, const Qinfo* q, double v );
		double vGetN( const Eref& e, const Qinfo* q ) const;
		void vSetNinit( const Eref& e, const Qinfo* q, double v );
		double vGetNinit( const Eref& e, const Qinfo* q ) const;
		void vSetDiffConst( const Eref& e, const Qinfo* q, double v );
		double vGetDiffConst( const Eref& e, const Qinfo* q ) const;

		void vSetConc( const Eref& e, const Qinfo* q, double v );
		double vGetConc( const Eref& e, const Qinfo* q ) const;
		void vSetConcInit( const Eref& e, const Qinfo* q, double v );
		double vGetConcInit( const Eref& e, const Qinfo* q ) const;

		void vSetSize( const Eref& e, const Qinfo* q, double v );
		double vGetSize( const Eref& e, const Qinfo* q ) const;

		void vSetSpecies( const Eref& e, const Qinfo* q, unsigned int v );
		unsigned int vGetSpecies( const Eref& e, const Qinfo* q ) const;

		//////////////////////////////////////////////////////////////////
		// Dest funcs
		//////////////////////////////////////////////////////////////////

		// void vProcess( const Eref& e, ProcPtr p );
		// void vReinit( const Eref& e, ProcPtr p );
		// void vReac( double A, double B );
		// void vHandleMolWt( const Eref& e, const Qinfo* q, double v );
		void vRemesh( const Eref& e, const Qinfo* q, 
			double oldvol,
			unsigned int numTotalEntries, unsigned int startEntry, 
			const vector< unsigned int >& localIndices, 
			const vector< double >& vols );

		//////////////////////////////////////////////////////////////////
		// utility funcs
		//////////////////////////////////////////////////////////////////
		void setSolver( Id solver );

		static const Cinfo* initCinfo();
	protected:
		StoichPools* stoich_;
};

#endif	// _Z_POOL_H

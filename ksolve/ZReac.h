/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _Z_REAC_H
#define _Z_REAC_H

class ZReac: public ReacBase
{
	public: 
		ZReac();
		~ZReac();

		//////////////////////////////////////////////////////////////////
		// Field assignment stuff
		//////////////////////////////////////////////////////////////////

		void vSetNumKf( const Eref& e, const Qinfo* q, double v );
		double vGetNumKf( const Eref& e, const Qinfo* q ) const;
		void vSetNumKb( const Eref& e, const Qinfo* q, double v );
		double vGetNumKb( const Eref& e, const Qinfo* q ) const;

		void vSetConcKf( const Eref& e, const Qinfo* q, double v );
		double vGetConcKf( const Eref& e, const Qinfo* q ) const;
		void vSetConcKb( const Eref& e, const Qinfo* q, double v );
		double vGetConcKb( const Eref& e, const Qinfo* q ) const;

		//////////////////////////////////////////////////////////////////
		// Dest funcs
		//////////////////////////////////////////////////////////////////
		void vRemesh( const Eref& e, const Qinfo* q );

		//////////////////////////////////////////////////////////////////
		// utility funcs
		//////////////////////////////////////////////////////////////////
		ZeroOrder* makeHalfReaction( 
			Element* orig, double rate, const SrcFinfo* finfo ) const;

		void setSolver( Id solver, Id orig );

		static const Cinfo* initCinfo();
	private:
		SolverBase* solver_;
};

#endif	// _Z_REAC_H

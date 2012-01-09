/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _ZOMBIE_REAC_H
#define _ZOMBIE_REAC_H

class ZombieReac
{
	public: 
		ZombieReac();
		~ZombieReac();

		//////////////////////////////////////////////////////////////////
		// Field assignment stuff
		//////////////////////////////////////////////////////////////////

		void setNumKf( const Eref& e, const Qinfo* q, double v );
		double getNumKf( const Eref& e, const Qinfo* q ) const;
		void setNumKb( const Eref& e, const Qinfo* q, double v );
		double getNumKb( const Eref& e, const Qinfo* q ) const;

		void setConcKf( const Eref& e, const Qinfo* q, double v );
		double getConcKf( const Eref& e, const Qinfo* q ) const;
		void setConcKb( const Eref& e, const Qinfo* q, double v );
		double getConcKb( const Eref& e, const Qinfo* q ) const;

		unsigned int getNumSub( const Eref& e, const Qinfo* q ) const;
		unsigned int getNumPrd( const Eref& e, const Qinfo* q ) const;

		//////////////////////////////////////////////////////////////////
		// Dest funcs
		//////////////////////////////////////////////////////////////////

		void process( const Eref& e, ProcPtr p );
		void reinit( const Eref& e, ProcPtr p );
		void sub( double v );
		void prd( double v );

		//////////////////////////////////////////////////////////////////
		// utility funcs
		//////////////////////////////////////////////////////////////////
		ZeroOrder* makeHalfReaction( 
			Element* orig, double rate, const SrcFinfo* finfo ) const;

		static void zombify( Element* solver, Element* orig );
		static void unzombify( Element* zombie );

		static const Cinfo* initCinfo();
	private:
		Stoich* stoich_;
		double concKf_;
		double concKb_;
};

#endif	// _ZOMBIE_REAC_H

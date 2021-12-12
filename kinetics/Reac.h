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
		virtual ~Reac();


		//////////////////////////////////////////////////////////////////
		// Field assignment stuff
		//////////////////////////////////////////////////////////////////

		void setNumKf( const Eref&e, double v );
		double getNumKf( const Eref& e ) const;
		void setNumKb( const Eref&e, double v );
		double getNumKb( const Eref& e ) const;

		/// set Kf in concentration units
		void setConcKf( const Eref& e, double v );
		/// get Kf in concentration units
		double getConcKf( const Eref& e ) const;

		/// set Kb in concentration units
		void setConcKb( const Eref& e, double v );
		/// get Kb in concentration units
		double getConcKb( const Eref& e ) const;

		/// Get number of substrates
		unsigned int getNumSub( const Eref& e ) const;
		/// Get number of products
		unsigned int getNumPrd( const Eref& e ) const;

		/// Look up the ObjId of the parent compartment of the reac.
		ObjId getCompartment( const Eref& e ) const;

		//////////////////////////////////////////////////////////////////
		// Dest funcs
		//////////////////////////////////////////////////////////////////

		/*
		void process( const Eref& e, ProcPtr p );
		void reinit( const Eref& e, ProcPtr p );
		void remesh( const Eref& e );
		*/
		void sub( double v );
		void prd( double v );
		void setSolver( const Eref& e, ObjId stoich );
		//////////////////////////////////////////////////////////////////

		static const Cinfo* initCinfo();
	private:
		double concKf_;	// Kf in concentration and time units
		double concKb_;	// Kb in concentration and time units
		Stoich* stoich_; // Stoich ptr defaults to 0.
};

#endif // REAC_H

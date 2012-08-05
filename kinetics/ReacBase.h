/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _REAC_BASE_H
#define _REAC_BASE_H

class ReacBase
{
	public: 
		ReacBase();
		virtual ~ReacBase();


		//////////////////////////////////////////////////////////////////
		// Field assignment stuff
		//////////////////////////////////////////////////////////////////

		void setNumKf( const Eref&e, const Qinfo* q, double v );
		double getNumKf( const Eref& e, const Qinfo *q ) const;
		void setNumKb( const Eref&e, const Qinfo* q, double v );
		double getNumKb( const Eref& e, const Qinfo *q ) const;

		/// set Kf in concentration units
		void setConcKf( const Eref& e, const Qinfo* q, double v );
		/// get Kf in concentration units
		double getConcKf( const Eref& e, const Qinfo* q ) const;

		/// set Kb in concentration units
		void setConcKb( const Eref& e, const Qinfo* q, double v );
		/// get Kb in concentration units
		double getConcKb( const Eref& e, const Qinfo* q ) const;

		/// Get number of substrates
		unsigned int getNumSub( const Eref& e, const Qinfo* q ) const;
		/// Get number of products
		unsigned int getNumPrd( const Eref& e, const Qinfo* q ) const;
		//////////////////////////////////////////////////////////////////
		// Inner virtual funcs for all the fields.
		virtual void vSetNumKf( const Eref&e, const Qinfo* q, double v ) =0;
		virtual double vGetNumKf( const Eref& e, const Qinfo *q ) const = 0;
		virtual void vSetNumKb( const Eref&e, const Qinfo* q, double v ) =0;
		virtual double vGetNumKb( const Eref& e, const Qinfo *q ) const = 0;
		virtual void vSetConcKf( const Eref& e, const Qinfo* q, double v )
			=0;
		virtual double vGetConcKf( const Eref& e, const Qinfo* q ) const = 0;
		virtual void vSetConcKb( const Eref& e, const Qinfo* q, double v ) = 0;
		virtual double vGetConcKb( const Eref& e, const Qinfo* q ) const = 0;
		// virtual unsigned int vGetNumSub( const Eref& e, const Qinfo* q ) const = 0;
		// virtual unsigned int vGetNumPrd( const Eref& e, const Qinfo* q ) const = 0;

		//////////////////////////////////////////////////////////////////
		/**
		 * Zombification functions.
		 */
		static void zombify( Element* original, const Cinfo* zClass, 
						Id solver );
		/// Assign solver info
		virtual void setSolver( Id solver, Id orig );

		//////////////////////////////////////////////////////////////////
		// Dest funcs
		//////////////////////////////////////////////////////////////////

		void process( const Eref& e, ProcPtr p );
		void reinit( const Eref& e, ProcPtr p );
		void sub( double v );
		void prd( double v );
		void remesh( const Eref& e, const Qinfo* q );
		//////////////////////////////////////////////////////////////////
		// Inner virtual funcs for the dest operations. Mostly do-nothing.
		virtual void vProcess( const Eref& e, ProcPtr p );
		virtual void vReinit( const Eref& e, ProcPtr p );
		virtual void vSub( double v );
		virtual void vPrd( double v );
		virtual void vRemesh( const Eref& e, const Qinfo* q );
		//////////////////////////////////////////////////////////////////

		static const Cinfo* initCinfo();
	protected:
		double concKf_;	// Kf in concentration and time units
		double concKb_;	// Kb in concentration and time units
};

#endif // REAC_BASE_H

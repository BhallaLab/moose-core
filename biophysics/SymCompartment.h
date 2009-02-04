/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _SYM_COMPARTMENT_H
#define _SYM_COMPARTMENT_H

/**
 * The Compartment class sets up an asymmetric compartment for
 * branched nerve calculations. Handles electronic structure and
 * also channels. This is not a particularly efficient way of doing
 * this, so we should use a solver for any substantial calculations.
 */
class SymCompartment: public moose::Compartment
{
	public:
			SymCompartment();
			~SymCompartment() {;}

			// Dest function definitions.
			static void raxial2Func(const Conn* c, double Ra, double Vm);
			static void sumRaxial( const Conn* c, double Ra );
			static void sumRaxialRequest( const Conn* c );
			static void sumRaxial2( const Conn* c, double Ra );
			static void sumRaxial2Request( const Conn* c );

	private:
			// These functions override the virtual equivalents from the
			// Compartment.
			void innerReinitFunc( Eref e, ProcInfo p );
			void innerRaxialFunc( double Ra, double Vm );
			void innerInitFunc( Eref e, ProcInfo p );

			// These functions are new for the Symcompartment.
			void innerRaxial2Func( double Ra, double Vm );

			double coeff_;
			double coeff2_;
};

// Used by the solver
extern const Cinfo* initSymCompartmentCinfo();

#endif // _SYM_COMPARTMENT_H

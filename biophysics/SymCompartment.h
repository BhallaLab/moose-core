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
 * The SymCompartment class sets up a symmetric compartment for
 * branched nerve calculations. Handles electronic structure and
 * also channels. This version splits the Ra between either end of the
 * compartment and is hence slightly cleaner than the asymmetric 
 * compartment.
 * The default EE method is not a particularly efficient way of doing
 * the calculations, so we should use a solver for any substantial 
 * calculations.
 */
class SymCompartment: public moose::Compartment
{
	public:
			SymCompartment();
			~SymCompartment() {;}

			// Dest function definitions.
                        void raxialSphere( double Ra, double Vm );
			void raxialSym( double Ra, double Vm );
			void sumRaxial( double Ra );
			/* void handleSumRaxialRequest( const Eref& e, const Qinfo* q ); */

			/* void raxial2Sym(double Ra, double Vm); */
			/* void sumRaxial2( double Ra ); */
			/* void handleSumRaxial2Request( const Eref& e, const Qinfo* q ); */

			static const Cinfo* initCinfo();

			// These functions override the virtual equivalents from the
			// Compartment.
			void innerReinit( const Eref& e, ProcPtr p );
			void innerInitProc( const Eref& e, ProcPtr p );
			void innerInitReinit( const Eref& e, ProcPtr p );
                        /* void process( const Eref& e, ProcPtr p ); */
	private:
                        // used for storing multiplicative coefficient computed from adjacent nodes in star-mesh transformation
			double coeff_;
			/* double coeff2_; */
			double RaSum_;
			/* double RaSum2_; */
};

#endif // _SYM_COMPARTMENT_H

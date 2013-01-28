/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _ODE_SYSTEM_H
#define _ODE_SYSTEM_H

class OdeSystem {
	public:
		OdeSystem();

		/**
		 * Creates a derived ODE system designed for a 
		 * junction with the specified compartment signature.
		 */
		OdeSystem( const StoichCore* master, 
						const vector< Id >& compartmentSignature );
		~OdeSystem();
		void reallyFreeOdeSystem();
		string setMethod( const string& method );

		void reinit( 
			void* gslStoich,
			int func (double t, const double *y, double *f, void *params),
			unsigned int nVarPools, double absAccuracy, double relAccuracy
		);

		/////////////////////////////////////////
		StoichCore* stoich_;

		/**
		 * This is the signature of OdeSystem in terms of which compartments
		 * it has cross-reactions with. If there are no entries, then it
		 * is completely self-contained. If there is one entry then this
		 * system is used only in cases abutting one other compartment
		 * with which there are cross-reactions. 
		 * Typically a given voxel will abut only a couple of compartments,
		 * but we may have to deal with the case of a single voxel which
		 * talks to a multitude of other compartments. So there is no
		 * limit to the length of this vector.
		 * In order to make the signature easily tested, it is sorted by
		 * numerical value of compartment Id.
		 */
		vector< Id > compartmentSignature_;
		// GSL stuff
		const gsl_odeiv_step_type* gslStepType_;
		gsl_odeiv_step* gslStep_;
		gsl_odeiv_control* gslControl_;
		gsl_odeiv_evolve* gslEvolve_;
		gsl_odeiv_system gslSys_;
};

#endif // _ODE_SYSTEM_H

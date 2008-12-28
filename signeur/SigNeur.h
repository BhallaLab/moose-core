/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _SIGNEUR_H
#define _SIGNEUR_H

/**
 * Utility function for getting dimensions of electrical compt when it
 * is divided up into numSeg.
 */
void getSigComptSize( const Eref& compt, unsigned int numSeg,
	double& volume, double& xByL );

/**
 * SigNeur: A multiscale model builder, aka a robot.
 * This potentially interfaces 4 models: An electrical cell model, and
 * chemical models for spines, dendrites and soma. The latter may be
 * 3-D stochastic, ODE, and genetic respectively, but for now the
 * class only handles ODE versions.
 *
 * Constructs a copy of the cell model on itself. The cell has its own
 * cell manager. Then there is an array of dends, of spines and 
 * soma models. Each of these incorporates diffusion, which may be
 * between categories. The whole mess sits on a kinetic manager as it
 * will start out by solving it using rk5. A 300-compartment model will
 * have ~50,000 molecular species. This is about 10 times the biggest I
 * have tried so far.
 * There is also an array of adaptors, one per cell model compt.
 * In due course need to put a distinct kind of solver for the spines,
 * either 3D monte carlo or Gillespie. Fortunately these will be separated
 * and hence can run on distinct processors.
 */
class SigNeur
{
	public:
		SigNeur();
		
		///////////////////////////////////////////////////
		// Field assignment functions
		///////////////////////////////////////////////////

		/// Timestep for signaling model. Also for flux and el interface
		static void setSigDt( const Conn* c, double value );
		static double getSigDt( Eref e );

		/// Timestep for Electrical model
		static void setCellDt( const Conn* c, double value );
		static double getCellDt( Eref e );

		/// Diffusion constant scaling factor, to globally modify diffusion
		static void setDscale( const Conn* c, double value );
		static double getDscale( Eref e );

		/// Options for parallel configuration. 
		static void setParallelMode( const Conn* c, int value );
		static int getParallelMode( Eref e );


		
		///////////////////////////////////////////////////
		// Dest function definitions
		///////////////////////////////////////////////////
		
		static void build( const Conn* c );
		void innerBuild( const Conn* c );

		///////////////////////////////////////////////////
		// Setup function definitions
		///////////////////////////////////////////////////
		bool traverseCell( Eref me );
		void schedule( Eref me );
		Id findSoma( const vector< Id >& compts );
		void buildTree( Id soma, const vector< Id >& compts );
		// void innerBuildTree( unsigned int parent, Eref e, int msg );
		void innerBuildTree( unsigned int parent, Eref paE, Eref e, 
			int msg1, int msg2 );

		/**
 		 * This function copies a signaling model. It first traverses
		 * the model and inserts any required diffusion reactions into
		 * the model. These are created as children of the molecule
		 * that diffuses, and are connected up locally for one-half of
		 * the diffusion process. Subsequently the system needs to 
		 * connect up to the next compartment, to set up the 
 		 * other half of the diffusion. Also the last diffusion reaction
 		 * needs to have its rates nulled out.
 		 *
 		 * Returns the root element of the copy.
 		 * Kinid is destination of copy
 		 * proto is prototype
 		 * Name is the name to assign to the copy.
 		 * num is the number of duplicates needed.
 		 */
		Element* copySig( Id kinId, Id proto, 
			const string& name, unsigned int num );
		/**
 		 * This variant of copySig makes multiple copies of a signaling
		 * model, but does NOT place them into an array. This is a 
		 * temporary * work-around necessitated because solvers don't 
		 * know how to deal with parts of arrays. The base element of
		 * the whole mess is a neutral so that there is a single 
		 * handle for the next stage of operations.
 		 * I would have preferred an array KineticManager, but that 
		 * gets messy.
 		 */ 
		Element* separateCopySig( Id kinId, Id proto, 
			const string& name, unsigned int num );
	
	private:
		/**
		 * sigDt_ is the timestep for signaling. This should be as long
		 * as possible, because most signaling solvers use adaptive 
		 * timesteps. The length of sigDt_ should be limited only by
		 * how often you want to read out, and how often you want external
		 * input to affect it.
		 */
		double sigDt_;	
					
		/**
		 * cellDt_ is the timestep for the compartmental biophysics solver.
		 * This is typically 50 to 100 usec. We currently do not have
		 * variable dt solvers, but that may change. In that case we
		 * may be able to stretch this time too.
		 */
		double cellDt_;

		double Dscale_;	/// Diffusion scale factor.

		unsigned int parallelMode_; 
};

#endif // _KINETIC_MANAGER_H

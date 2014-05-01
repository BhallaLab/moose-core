/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2013 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _DSOLVE_H
#define _DSOLVE_H

/**
 * The Dsolve manages a large number of pools, each inhabiting a large
 * number of voxels that are shared for all the pools. 
 * Each pool is represented by an array of concs, one for each voxel.
 * Each such array is kept on a single node for efficient solution.
 * The different pool arrays are assigned to different nodes for balance.
 * All pool arrays 
 * We have the parent Dsolve as a global. It constructs the diffusion
 * matrix from the NeuroMesh and generates the opvecs.
 * We have the child DiffPoolVec as a local. Each one contains a
 * vector of pool 'n' in each voxel, plus the opvec for that pool. There
 * is an array of DiffPoolVecs, one for each species, and we let the
 * system put each DiffPoolVec on a suitable node for balancing.
 * Some DiffPoolVecs are for molecules that don't diffuse. These
 * simply have an empty opvec.
 */
class Dsolve: public ZombiePoolInterface
{
	public: 
		Dsolve();
		~Dsolve();

		//////////////////////////////////////////////////////////////////
		// Field assignment stuff
		//////////////////////////////////////////////////////////////////
		unsigned int getNumVarPools() const;

		void setStoich( const Eref& e, Id id );
		Id getStoich( const Eref& e ) const;
		void setCompartment( Id id );
		Id getCompartment() const;

		void setPath( const Eref& e, string path );
		string getPath( const Eref& e ) const;

		unsigned int getNumVoxels() const;

		vector< double > getNvec( unsigned int pool ) const;
		void setNvec( unsigned int pool, vector< double > vec );

		//////////////////////////////////////////////////////////////////
		// Dest Finfos
		//////////////////////////////////////////////////////////////////
		void process( const Eref& e, ProcPtr p );
		void reinit( const Eref& e, ProcPtr p );

		//////////////////////////////////////////////////////////////////
		// Inherited virtual funcs from ZombiePoolInterface
		//////////////////////////////////////////////////////////////////
		double getNinit( const Eref& e ) const;
		void setNinit( const Eref& e, double value );
		double getN( const Eref& e ) const;
		void setN( const Eref& e, double value );
		double getDiffConst( const Eref& e ) const;
		void setDiffConst( const Eref& e, double value );
		void setMotorConst( const Eref& e, double value );

		void setNumPools( unsigned int num );
		unsigned int getNumPools() const;

		void getBlock( vector< double >& values ) const;
		void setBlock( const vector< double >& values );

		//////////////////////////////////////////////////////////////////
		// Model traversal and building functions
		//////////////////////////////////////////////////////////////////
		unsigned int convertIdToPoolIndex( const Eref& e ) const;

		/** 
		 * Fills in poolMap_ using elist of objects found when the 
		 * 'setPath' function is executed. temp is returned with the
		 * list of PoolBase objects that exist on the path.
		 */
		void makePoolMapFromElist( const vector< ObjId >& elist,
						vector< Id >& temp );

		/**
		 * Does nasty message traversal to look up the clock tick that
		 * sends the Process/reinit message to the Dsolve (specified by e)
		 * and then figure out the dt used.
		 */
		double findDt( const Eref& e );

		/** 
		 * This key function does the work of setting up the Dsolve. 
		 * Should be called after the compartment has been attached to
		 * the Dsolve, and the stoich is assigned.
		 * Called during the setStoich function.
		 */
		void build( double dt );
		void rebuildPools();

		/**
		 * Utility func for debugging: Prints N_ matrix
		 */
		void print() const;

		//////////////////////////////////////////////////////////////////
		static const Cinfo* initCinfo();
	private:
		/**
		 * Id of compartment object that specifies the diffusion space
		 * of the Dsolve. Must be either a CylMesh or a NeuroMesh.
		 */
		Id compartment_;

		/** 
		 * Id of stoich object for reac system whose pools are being
		 * diffused by this Dsolve. Optional, the path below can also be
		 * used.
		 */
		Id stoich_;

		/// Path of pools managed by Dsolve, may include other classes too.
		string path_; 

		unsigned int numTotPools_;
		unsigned int numLocalPools_;
		unsigned int poolStartIndex_;
		unsigned int numVoxels_;

		/// Internal vector, one for each pool species managed by Dsolve.
		vector< DiffPoolVec > pools_;

		/// smallest Id value for poolMap_
		unsigned int poolMapStart_;

		/// Looks up pool# from pool Id, using poolMapStart_ as offset.
		vector< unsigned int > poolMap_;
};


#endif	// _DSOLVE_H

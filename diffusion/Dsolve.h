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

		void setStoich( Id id );
		Id getStoich() const;
		void setCompartment( Id id );
		Id getCompartment() const;

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

		void setNumPools( unsigned int num );
		unsigned int getNumPools() const;

		//////////////////////////////////////////////////////////////////
		// Model traversal and building functions
		//////////////////////////////////////////////////////////////////
		unsigned int convertIdToPoolIndex( const Eref& e ) const;

		// This key function does the work. Should be called after 
		// all the stoich and compartment stuff is assigned.
		void build();

		/**
		 * Utility func for debugging: Prints N_ matrix
		 */
		void print() const;

		//////////////////////////////////////////////////////////////////
		static const Cinfo* initCinfo();
	private:
		Id compartment_;
		Id stoich_;

		unsigned int numTotPools_;
		unsigned int numLocalPools_;
		unsigned int poolStartIndex_;
		unsigned int numVoxels_;
		vector< DiffPoolVec > pools_;

		/// smallest Id value for pools managed by Dsolve. Used for lookup.
		unsigned int poolMapStart_;

		/// Looks up pool# from pool Id, using poolMapStart_ as offset.
		vector< unsigned int > poolMap_;
};


#endif	// _DSOLVE_H

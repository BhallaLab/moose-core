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
class Dsolve
{
	public: 
		Dsolve();
		~Dsolve();

		//////////////////////////////////////////////////////////////////
		// Field assignment stuff
		//////////////////////////////////////////////////////////////////
		unsigned int getNumVarPools() const;

		void setPath( const Eref& e, string v );
		string getPath( const Eref& e ) const;

		unsigned int getNumVoxels() const;

		//////////////////////////////////////////////////////////////////
		// Dest funcs
		//////////////////////////////////////////////////////////////////

		//////////////////////////////////////////////////////////////////
		// Model traversal and building functions
		//////////////////////////////////////////////////////////////////

		/**
		 * zombifyModel marches through the specified id list and 
		 * converts all entries into zombies. The first arg e is the
		 * Eref of the Stoich itself.
		 */
		void zombifyModel( const Eref& e, const vector< Id >& elist );

		/**
		 * Converts back to ExpEuler type basic kinetic Elements.
		 */
		void unZombifyModel();


		/**
		 * Utility func for debugging: Prints N_ matrix
		 */
		void print() const;

#ifdef USE_GSL
		static int gslFunc( double t, const double* y, double* yprime, void* s );
		int innerGslFunc( double t, const double* y, double* yprime,
			unsigned int meshIndex );
#endif // USE_GSL


		//////////////////////////////////////////////////////////////////
		static const Cinfo* initCinfo();
	private:
		string path_;
		vector< DiffPoolVec > pools_;
};


#endif	// _DSOLVE_H

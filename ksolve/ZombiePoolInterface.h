/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2014 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _ZOMBIE_POOL_INTERFACE_H
#define _ZOMBIE_POOL_INTERFACE_H

/**
 * This pure virtual base class is for solvers that want to talk to
 * the zombie pool. 
 * The Eref specifies both the pool identity and the voxel number within
 * the pool.
 */
class ZombiePoolInterface
{
	public:
		/// Set initial # of molecules in given pool and voxel. Bdry cond.
		virtual void setNinit( const Eref& e, double val ) = 0;
		/// get initial # of molecules in given pool and voxel. Bdry cond.
		virtual double getNinit( const Eref& e ) const = 0;

		/// Set # of molecules in given pool and voxel. Varies with time.
		virtual void setN( const Eref& e, double val ) = 0;
		/// Get # of molecules in given pool and voxel. Varies with time.
		virtual double getN( const Eref& e ) const = 0;

		/// Diffusion constant: Only one per pool, voxel number is ignored.
		virtual void setDiffConst( const Eref& e, double val ) = 0;
		/// Diffusion constant: Only one per pool, voxel number is ignored.
		virtual double getDiffConst( const Eref& e ) const = 0;

		/// Specifies number of pools (species) handled by system.
		virtual void setNumPools( unsigned int num ) = 0;
		/// gets number of pools (species) handled by system.
		virtual unsigned int getNumPools() const = 0;
};

#endif	// _ZOMBIE_POOL_INTERFACE_H

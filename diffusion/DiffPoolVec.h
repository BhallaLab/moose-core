/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2012 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _DIFF_POOL_VEC_H
#define _DIFF_POOL_VEC_H

/**
 * This is a FieldElement of the Dsolve class. It manages (ie., zombifies)
 * a specific pool, and the pool maintains a pointer to it. For accessing
 * volumes, this maintains a pointer to the relevant ChemCompt.
 */
class DiffPoolVec: public ZombiePoolInterface
{
	public:
		DiffPoolVec();
		void process();
		void reinit();
		void advance();
		// Inherited virtual funcs from ZombiePoolInterface
		double getNinit( const Eref& e ) const;
		void setNinit( const Eref& e, double value );
		double getN( const Eref& e ) const;
		void setN( const Eref& e, double value );
		double getDiffConst() const;
		void setDiffConst( double value );

		/////////////////////////////////////////////////
		Id getPool() const; /// Returns pool.
		void setPool( Id pool ); /// Assigns pool id.
		vector< double >& n(); /// Used by parent solver to manipulate 'n'
		void setOps( const vector< Triplet< double > >& ops_, 
				const vector< double >& diagVal_ ); /// Assign operations.

		// static const Cinfo* initCinfo();
	private:
		Id pool_;	/// Specifies the pool to which this is attached.
		Id parent_;	/// Specifies the parent Dsolve.
		vector< double > n_; /// Number of molecules of pool in each voxel
		vector< double > nInit_; /// Boundary condition: Initial 'n'.
		double diffConst_; /// Diffusion const, assumed uniform
		vector< Triplet< double > > ops_;
		vector< double > diagVal_;
};

#endif // _DIFF_POOL_VEC_H

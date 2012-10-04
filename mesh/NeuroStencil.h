/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2012 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _NEURO_STENCIL_H
#define _NEURO_STENCIL_H

class NeuroStencil: public Stencil
{
	public:
		NeuroStencil( const vector< NeuroNode >& nodes, 
					const vector< unsigned int >& nodeIndex,
					const vector< double >& vs,
					const vector< double >& area);
		NeuroStencil();

		~NeuroStencil();

		/**
		 * computes the Flux f in the voxel on meshIndex. Takes the
		 * matrix of molNumber[meshIndex][pool] and 
		 * the vector of diffusionConst[pool] as arguments.
		 */
		void addFlux( unsigned int meshIndex, 
			vector< double >& f, const vector< vector< double > >& S, 
			const vector< double >& diffConst ) const;

		void addLinearFlux( unsigned int index, 
			vector< double >& f, 
			const vector< double >& tminus,
			const vector< double >& t0,
			const vector< double >& tplus,
			double aminus,
			double aplus,
			double vsminus,
			double vs0,
			double vsplus,
			double invSq,
			const vector< double >& diffConst ) const;

		void addHalfFlux( unsigned int index, 
			vector< double >& f, 
			const vector< double >& t0,
			const vector< double >& tprime,
			double area,
			double vs0,
			double vsprime,
			double invSq,
			const vector< double >& diffConst ) const;

	private:
		const vector< NeuroNode >& nodes_;
		const vector< unsigned int >& nodeIndex_;
		const vector< double >& vs_;
		const vector< double >& area_;

		/**
		 * This stores the effective length between nodes as a sparse
		 * matrix. Most node pairs are not coupled so we leave them out.
		 * L_[voxelIndex][voxelIndex]
		 * This is actually a symmetric matrix, in due course would like
		 * to implement a symmetric sparse matrix.
		 * Effective length is computed as 
		 * Lij = 0.5* (Lmax * Amin / Amax + Lmin )
		 * where Lmin and Amin are length and area of the smaller diameter
		 * voxel.
		 */
		SparseMatrix< double > L_;

};

#endif // _NEURO_STENCIL_H

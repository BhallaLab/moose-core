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
		NeuroStencil();

		~NeuroStencil();

		void setNodes( const vector< NeuroNode >* nodes );
		void setNodeIndex( const vector< unsigned int >* nodeIndex );

		/**
		 * computes the Flux f in the voxel on meshIndex. Takes the
		 * matrix of molNumber[meshIndex][pool] and 
		 * the vector of diffusionConst[pool] as arguments.
		 */
		void addFlux( unsigned int meshIndex, 
			vector< double >& f, const vector< vector< double > >& S, 
			const vector< double >& diffConst ) const;

	private:
		const vector< NeuroNode >* nodes_;
		const vector< unsigned int >* nodeIndex_;
};

#endif // _NEURO_STENCIL_H

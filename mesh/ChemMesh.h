/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _CHEM_MESH_H
#define _CHEM_MESH_H

/**
 * The ChemMesh represents a chemically identified compartment.
 * This may be spatially extended, and may even be discontinuous.
 * The same set of reactions and molecules populates any given compartment.
 * Examples of compartments might be: nucleus, cell membrane, 
 * early endosomes, spine heads.
 * Connects to one or more 'Geometry' elements to define its boundaries.
 */
class ChemMesh
{
	public: 
		ChemMesh();
		virtual ~ChemMesh();
		//////////////////////////////////////////////////////////////////
		// Field assignment stuff
		//////////////////////////////////////////////////////////////////

		double getEntireSize() const;

		unsigned int getDimensions() const;
		virtual unsigned int innerGetDimensions() const = 0;

		//////////////////////////////////////////////////////////////////
		// FieldElementFinfo stuff for MeshEntry lookup
		//////////////////////////////////////////////////////////////////
		/**
		 * Returns the number of MeshEntries on this ChemMesh
		 */
		unsigned int getNumEntries() const;
		virtual unsigned int innerNumEntries() const = 0;

		/**
		 * Dummy function. The numEntries is only set by other functions
		 * that define compartment decomposition.
		 */
		void setNumEntries( unsigned int num );
		virtual void innerSetNumEntries( unsigned int n ) = 0;

		/**
		 * Returns the matched lookupEntry
		 */
		MeshEntry* lookupEntry( unsigned int index );
		//////////////////////////////////////////////////////////////////
		// FieldElement assignment stuff for MeshEntries
		//////////////////////////////////////////////////////////////////
		/// Virtual function to return MeshType of specified entry.
		virtual unsigned int getMeshType( unsigned int fid )
			const = 0;
		/// Virtual function to return dimensions of specified entry.
		virtual unsigned int getMeshDimensions( unsigned int fid )
			const = 0;
		/// Virtual function to return volume of mesh Entry.
		virtual double getMeshEntrySize( unsigned int fid ) 
			const = 0;
		/// Virtual function to return coords of mesh Entry.
		virtual vector< double > getCoordinates( unsigned int fid ) 
			const = 0;
		/// Virtual function to return info on Entries connected to this one
		virtual vector< unsigned int > getNeighbors( unsigned int fid ) 
			const = 0;
		/// Virtual function to return diffusion X-section area per neighbor
		virtual vector< double > getDiffusionArea( unsigned int fid )
			const = 0;
		/// Virtual function to return scale factor for diffusion. 1 here.
		virtual vector< double > getDiffusionScaling( unsigned int fid ) 
			const = 0;

		//////////////////////////////////////////////////////////////////

		static const Cinfo* initCinfo();

	protected:
		double size_; /// Volume or area
	private:
		MeshEntry entry_; // Wrapper for self ptr
};

#endif	// _CHEM_MESH_H

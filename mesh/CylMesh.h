/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2011 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _CYL_MESH_H
#define _CYL_MESH_H

/**
 * The CylMesh represents a chemically identified compartment shaped
 * like an extended cylinder. This is psuedo-1 dimension: Only the
 * axial dimension is considered for diffusion and subdivisions.
 * Typically used in modelling small segments of dendrite
 */
class CylMesh: public ChemMesh
{
	public: 
		CylMesh();
		~CylMesh();
		//////////////////////////////////////////////////////////////////
		//  Utility func
		//////////////////////////////////////////////////////////////////
		/**
		 * Recomputes all local coordinate and meshing data following
		 * a change in any of the coord parameters
		 */

		void updateCoords();

		//////////////////////////////////////////////////////////////////
		// Field assignment stuff
		//////////////////////////////////////////////////////////////////

		void setX0( double v );
		double getX0() const;
		void setY0( double v );
		double getY0() const;
		void setZ0( double v );
		double getZ0() const;
		void setR0( double v );
		double getR0() const;

		void setX1( double v );
		double getX1() const;
		void setY1( double v );
		double getY1() const;
		void setZ1( double v );
		double getZ1() const;
		void setR1( double v );
		double getR1() const;

		void setCoords( vector< double > v );
		vector< double > getCoords() const;

		void setLambda( double v );
		double getLambda() const;

		double getTotLength() const;

		unsigned int innerGetDimensions() const;

		//////////////////////////////////////////////////////////////////
		// FieldElement assignment stuff for MeshEntries
		//////////////////////////////////////////////////////////////////

		/// Virtual function to return MeshType of specified entry.
		unsigned int getMeshType( unsigned int fid ) const;
		/// Virtual function to return dimensions of specified entry.
		unsigned int getMeshDimensions( unsigned int fid ) const;
		/// Virtual function to return volume of mesh Entry.
		double getMeshEntrySize( unsigned int fid ) const;
		/// Virtual function to return coords of mesh Entry.
		vector< double > getCoordinates( unsigned int fid ) const;
		/// Virtual function to return info on Entries connected to this one
		vector< unsigned int > getNeighbors( unsigned int fid ) const;
		/// Virtual function to return diffusion X-section area
		vector< double > getDiffusionArea( unsigned int fid ) const;
		/// Virtual function to return scale factor for diffusion. 1 here.
		vector< double > getDiffusionScaling( unsigned int fid ) const;

		//////////////////////////////////////////////////////////////////

		/**
		 * Inherited virtual func. Returns number of MeshEntry in array
		 */
		unsigned int innerGetNumEntries() const;
		/// Inherited virtual func.
		void innerSetNumEntries( unsigned int n );
		
		//////////////////////////////////////////////////////////////////
		// Dest funcs
		//////////////////////////////////////////////////////////////////

		static const Cinfo* initCinfo();

	private:
		double size_; /// Total Volume
		unsigned int numEntries_; // Number of subdivisions to use
		bool useCaps_; // Flag: Should the ends have hemispherical caps?
		bool isToroid_; // Flag: Should the ends loop around mathemagically?

		double x0_; /// coords
		double y0_; /// coords
		double z0_; /// coords

		double x1_; /// coords
		double y1_; /// coords
		double z1_; /// coords

		double r0_;	/// Radius at one end
		double r1_; /// Radius at other end

		double lambda_;	/// Length constant for diffusion

		double totLen_;	/// Utility value: Total length of cylinder
		double rSlope_;	/// Utility value: dr/dx
		double lenSlope_; /// Utility value: dlen/dx


		/**
		 * For spherical mesh, coords are xyz r0 r1 theta0 theta1 phi0 phi1
		 * For Cylindrical mesh, coords are x1y1z1 x2y2z2 r0 r1 phi0 phi1
		 * For cuboid mesh, coords are x1y1z1 x2y2z2
		 * For tetrahedral mesh, coords are x1y1z1 x2y2z2 x3y3z3 x4y4z4
		 * Later worry about planar meshes. Those would be surfaces
		vector< double > coords_;
		 */

		/// Indices of neighbours in the current mesh array.
		// vector< unsigned int >neighbours; 

		/*
		mesh dimension:
		mesh type (one for each dim): none, axial, radial, 
		*/
};

#endif	// _CYL_MESH_H

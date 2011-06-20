/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2011 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _CUBE_MESH_H
#define _CUBE_MESH_H

/**
 * The CubeMesh represents a chemically identified compartment shaped
 * like an extended cylinder. This is psuedo-1 dimension: Only the
 * axial dimension is considered for diffusion and subdivisions.
 * Typically used in modelling small segments of dendrite
 */
class CubeMesh: public ChemMesh
{
	public: 
		CubeMesh();
		~CubeMesh();

		//////////////////////////////////////////////////////////////////
		// Field assignment stuff
		//////////////////////////////////////////////////////////////////

		void setX0( double v );
		double getX0() const;
		void setY0( double v );
		double getY0() const;
		void setZ0( double v );
		double getZ0() const;

		void setX1( double v );
		double getX1() const;
		void setY1( double v );
		double getY1() const;
		void setZ1( double v );
		double getZ1() const;

		void setDx( double v );
		double getDx() const;
		void setDy( double v );
		double getDy() const;
		void setDz( double v );
		double getDz() const;

		void setCoords( vector< double > v );
		vector< double > getCoords() const;

		void setMeshToSpace( vector< unsigned int > v );
		vector< unsigned int > getMeshToSpace() const;

		void setSpaceToMesh( vector< unsigned int > v );
		vector< unsigned int > getSpaceToMesh() const;

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

		void buildMesh( Id geom, double x, double y, double z );

		//////////////////////////////////////////////////////////////////
		//  Utility func
		//////////////////////////////////////////////////////////////////
		/**
		 * Recomputes all local coordinate and meshing data following
		 * a change in any of the coord parameters
		 */
		void updateCoords();

		unsigned int neighbor( unsigned int spaceIndex, 
			int dx, int dy, int dz ) const;


		//////////////////////////////////////////////////////////////////

		static const Cinfo* initCinfo();

	private:
		double size_; /// Total Volume
		bool isToroid_; // Flag: Should the ends loop around mathemagically?

		double x0_; /// coords
		double y0_; /// coords
		double z0_; /// coords

		double x1_; /// coords
		double y1_; /// coords
		double z1_; /// coords

		double dx_; /// Cuboid edge
		double dy_; /// Cuboid edge
		double dz_; /// Cuboid edge

		unsigned int nx_; /// # of entries in x in surround volume
		unsigned int ny_; /// # of entries in y in surround volume
		unsigned int nz_; /// # of entries in z in surround volume

		/**
		 * For spherical mesh, coords are xyz r0 r1 theta0 theta1 phi0 phi1
		 * For Cylindrical mesh, coords are x1y1z1 x2y2z2 r0 r1 phi0 phi1
		 * For cuboid mesh, coords are x1y1z1 x2y2z2
		 * For tetrahedral mesh, coords are x1y1z1 x2y2z2 x3y3z3 x4y4z4
		 * Later worry about planar meshes. Those would be surfaces
		 */

		vector< unsigned int > m2s_;
		vector< unsigned int > s2m_;
};

#endif	// _CUBE_MESH_H

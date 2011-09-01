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
 * like a cuboid. This is not really an effective geometry for most
 * neurons because it would have to be rather finely subdivided to fit
 * a typical dendrite or soma volume, but it is general.
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

		void setNx( unsigned int v );
		unsigned int getNx() const;
		void setNy( unsigned int v );
		unsigned int getNy() const;
		void setNz( unsigned int v );
		unsigned int getNz() const;

		void setCoords( vector< double > v );
		vector< double > getCoords() const;

		void setMeshToSpace( vector< unsigned int > v );
		vector< unsigned int > getMeshToSpace() const;

		void setSpaceToMesh( vector< unsigned int > v );
		vector< unsigned int > getSpaceToMesh() const;

		unsigned int innerGetDimensions() const;

		void setIsToroid( bool v );
		bool getIsToroid() const;

		void setPreserveNumEntries( bool v );
		bool getPreserveNumEntries() const;

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
		bool isToroid_; ///Flag: Should the ends loop around mathemagically?
		bool preserveNumEntries_; ///Flag: Should dx change or nx, with vol?

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

		/**
		 * Mesh to Space lookup. Indexed by linear mesh index, from 0 to
		 * number of actual mesh entries (occupied cuboids). Returns 
		 * spatial index, from 0 to nx * ny * nz - 1.
		 * Needed whenever the cuboid mesh is not filling the entire volume
		 * of the cube, that is, in most cases.
		 */
		vector< unsigned int > m2s_;

		/**
		 * Space to Mesh lookup. Indexed by spatial index, from
		 * 0 to nx * ny * nz - 1. Specifically, point x y z is indexed as
		 * ( z * ny + y ) * nx + x. Returns mesh index to look up molecules
		 * etc in the specific volume. In case the spatial location is 
		 * outside the included volume of the mesh, returns ~0.
		 */
		vector< unsigned int > s2m_;
};

#endif	// _CUBE_MESH_H

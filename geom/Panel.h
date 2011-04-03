/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/


#ifndef _PANEL_H
#define _PANEL_H

/**
 * The Panel corresponds to the Smoldyn panelstruct. Here we will
 * set up a Panel base class and derive specific shapes off it.
 * All the shapes use coord vectors and here we provide generic
 * access functions to them.
 */
class Panel
{
	public:
		Panel( unsigned int nDims = 3, unsigned int nPts = 3 );
		virtual ~Panel() {
			;
		}
		
		///////////////////////////////////////////////////
		// Field assignment functions
		///////////////////////////////////////////////////

		
		/// Returns number of defining pts for panel
		unsigned int getNpts() const;
		virtual unsigned int localNpts() const;

		/// Returns dimensions of panel
		unsigned int getNdims() const;

		/// Returns shape id, using the same fields of Smoldyn.
		unsigned int getShapeId() const;
		virtual unsigned int localShapeId() const;

		/// Returns number of neighbors of panel
		unsigned int getNumNeighbors( const Eref& e, const Qinfo* q ) const;

		/// Assigns X coordinate by index.
		void setX( unsigned int i, double val );
		double getX( unsigned int i ) const;

		/// Assigns Y coordinate by index.
		void setY( unsigned int i, double val );
		double getY( unsigned int i ) const;

		/// Assigns Z coordinate by index.
		void setZ( unsigned int i, double val );
		double getZ( unsigned int i ) const;

		/**
		 * Gets all the coordinates as a vector, with the entire X vector 
		 * first, then the Y vector, then Z. The vector lengths must match.
		 * If 2 dimensional, then the Z vector can be left out.
		 * If 1 dimensional, then the Z and Y vectors can be left out.
		 */
		vector< double > getCoords() const;

		/**
		 * Sets all the coordinates as a vector, with the entire X vector 
		 * first, then the Y vector, then Z.
		 */
		void setCoords( vector< double > v );


		/**
		 * For future use: Convert the current shape into a set of 
		 * triangular
		 * finite element vertices (assuming 3 d). The fineness of the
		 * grid is set by the 'area' argument.
		*/
		vector< double > getFiniteElementVertices( double area ) const;

		virtual void localFiniteElementVertices( 
			vector< double >& ret,  double area ) const;

		///////////////////////////////////////////////////////////////
		// Dest Funcs
		///////////////////////////////////////////////////////////////
		void handleNeighbor();

		///////////////////////////////////////////////////
		// Utility funcs
		///////////////////////////////////////////////////
		virtual void localSetPos( unsigned int index, unsigned int dim, 
			double value );

		virtual double localGetPos( unsigned int index, unsigned int dim ) const; 

		///////////////////////////////////////////////////////////////
		static const Cinfo* initCinfo();

	private:

		/**
		 * front is a term to indicate if
		 * the surface is parallel or antiparallel to the normal to the
		 * surface. Each surface has its own rule about the normal.
		 * Steve advises to hide.
		 */
		bool front_;	

		/**
		 * nDims is the number of dimensions of the panel. Usually 3,
		 * in fact at this point isn't likely to work for other dims.
		 */
		unsigned int nDims_;

		/**
		 * Vector of coords. coords_[number][dimension]
		 */
		vector< double > coords_;

};

// Protect it here from the Smoldyn enum with the same args.
namespace Moose {
	enum PanelShape {PSrect,PStri,PSsph,PScyl,PShemi,PSdisk,PSall,PSnone};
}

#endif // _PANEL_H

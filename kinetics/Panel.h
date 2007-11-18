/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/


#ifndef _Panel_h
#define _Panel_h

/**
 * The Panel corresponds to the Smoldyn panelstruct. Here we will
 * set up a Panel base class and derive specific shapes off it.
 * All the shapes use coord vectors and here we provide generic
 * access functions to them.
 */
class Panel
{
	public:
		Panel( unsigned int nDims = 3 );
		virtual ~Panel() {
			;
		}
		
		///////////////////////////////////////////////////
		// Field assignment functions
		///////////////////////////////////////////////////

		
		/// Returns number of defining pts for panel
		static unsigned int getNpts( const Element* e );
		virtual unsigned int localNpts() const;

		/// Returns dimensions of panel
		static unsigned int getNdims( const Element* e );

		/// Returns shape id, using the same fields of Smoldyn.
		static unsigned int getShapeId( const Element* e );
		virtual unsigned int localShapeId() const;

		/// Returns number of neighbors of panel
		static unsigned int getNneighbors( const Element* e );

		static void setPos( const Conn& c, double value, 
				unsigned int i, unsigned int dim );
		void localSetPos( double value, unsigned int i, unsigned int dim );

		static double getPos( const Element* e, unsigned int i, 
				unsigned int dim);
		double localGetPos( unsigned int i, unsigned int dim);

		static void setX( const Conn& c, double val, const unsigned int& i);
		static double getX( const Element* e, const unsigned int& i );
		static void setY( const Conn& c, double val, const unsigned int& i);
		static double getY( const Element* e, const unsigned int& i );
		static void setZ( const Conn& c, double val, const unsigned int& i);
		static double getZ( const Element* e, const unsigned int& i );


		/**
		 * For future use: Convert the current shape into a set of 
		 * triangular
		 * finite element vertices (assuming 3 d). The fineness of the
		 * grid is set by the 'area' argument.
		*/
		static vector< double > getFiniteElementVertices(
			const Element* e, double area );

	private:

		/**
		 * front is a sign term to indicate if the 'front' face of
		 * the surface is parallel or antiparallel to the normal to the
		 * surface. Each surface has its own rule about the normal.
		 */
		bool front_;	

		/**
		 * Vector of coords. coords_[number][dimension]
		 */
		vector< double > coords_;

		unsigned int nDims_;
};

// Used by the Smoldyn solver
extern const Cinfo* initPanelCinfo();

#endif // _Panel_h

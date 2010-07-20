/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/


#ifndef _KinCompt_h
#define _KinCompt_h

/**
 * The KinCompt is a compartment for kinetic calculations. It doesn't
 * really correspond to a single Smoldyn concept, but it encapsulates
 * many of them into the traditional compartmental view. It connects up
 * with one or more surfaces which collectively define its volume and
 * geometry.
 */
class KinCompt
{
	public:
		KinCompt();
		virtual ~KinCompt()
		{ ; }
		
		///////////////////////////////////////////////////
		// Field assignment functions
		///////////////////////////////////////////////////
		static double getVolume( Eref e );
		static void setVolume( const Conn* c, double value );
		static double getArea( Eref e );
		static void setArea( const Conn* c, double value );
		static double getPerimeter( Eref e );
		static void setPerimeter( const Conn* c, double value );
		static double getSize( Eref e );
		static void setSize( const Conn* c, double value );
		static void setSizeWithoutRescale( const Conn* c, double value );
		// This is specialized in the derived class KineticManager.
		virtual void innerSetSize( Eref e, double value, bool ignoreRescale = 0 );
		static unsigned int getNumDimensions( Eref e );
		static void setNumDimensions( const Conn* c, unsigned int value );
		static double getX( Eref e );
		static void setX( const Conn* c, double value );
		static double getY( Eref e );
		static void setY( const Conn* c, double value );
		///////////////////////////////////////////////////
		// Message handlers
		///////////////////////////////////////////////////
		static void requestExtent( const Conn* c );
		void innerRequestExtent( Eref e ) const;

		static void exteriorFunction( const Conn* c, 
			double v1, double v2, double v3 );
		void localExteriorFunction( double v1, double v2, double v3 );

		static void interiorFunction( const Conn* c, 
			double v1, double v2, double v3 );
		void localInteriorFunction( double v1, double v2, double v3 );

		static void rescaleFunction( const Conn* c, double ratio );

		/**
		 * Special MsgDest used by Molecules reading from kkit.
		 * Molecules try to set their volumes by referring to this
		 * KinCompt, which has to decide what to do about it depending
		 * on history of such assignments.
		 * If it is the first: Assign without rescaling
		 * If it is a later one, same vol: Just keep tally, silently.
		 * If it is a later one, new vol: Complain, tally
		 * If the later new vols outnumber original vol: Complain louder
		 */
		static void setVolumeFromChild( 
			const Conn* c, string ch, double v );
		void innerSetVolumeFromChild( Eref pa, string ch, double v );

	protected:
		double size() const {
			return size_;
		}
	private:

		/**
		 * Size is the variable-unit extent of the compartment. Its
		 * dimensions depend on the numDimensions of the compartment.
		 * So, when numDimensions == 3, size == volume.
		 * It is exactly equivalent to the size field in SBML
		 * compartments.
		 */
		double size_;

		/**
		 * Volume is computed by summing contributions from all surfaces.
		 * It first gets the exterior message from the outside surface.
		 * This assigns the volume, eliminating earlier values.
		 * Then the interior messages subtract from it.
		 */
		double volume_; 

		/**
		 * Surface area of compartment. Exterior message assigns it,
		 * then interior messages _add_ to it if numDimensions == 3,
		 * but _subtract_ from it if numDimensions == 2. Think about it.
		 */
		double area_; 
		
		/**
		 * Perimeter of compartment. Relevant for surfaces only.
		 */
		double perimeter_; 

		/**
		 * Number of dimensions represented by the compartment.
		 * 3 for normal compartments, 2 for membrane surfaces.
		 */
		unsigned int numDimensions_;

		/**
		 * Backward compat hack: Keeps track of number of molecules
		 * assigned to this KinCompt, and number whose volumes match up.
		 * The remainder would like some other volume.
		 */
		 unsigned int numAssigned_;
		 unsigned int numMatching_;

		double x_;		/// x coordinate for display
		double y_;		/// y coordinate for display

};

// Used by the Smoldyn solver
extern const Cinfo* initKinComptCinfo();

// Used by the KineticManager to handle rescaling.
extern void rescaleTree( Eref e, double ratio );

/// Used by KineticHub, Molecule, Reaction, and enzyme.
extern double getVolScale( Eref e );
extern void setParentalVolScale( Eref e, double volScale );

#endif // _KinCompt_h

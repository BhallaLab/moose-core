/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/


#ifndef _Geometry_h
#define _Geometry_h

/**
 * The Geometry corresponds to the Smoldyn surfacesuperstruct.
 * It manages multiple surfaces that matter to a given solver.
 * It mostly handles a list of surfaces, but has a couple of global control
 * parameters for tolerances.
 */
class Geometry
{
	public:
		Geometry();
		
		///////////////////////////////////////////////////
		// Field assignment functions
		///////////////////////////////////////////////////
		static void setEpsilon( const Conn* c, double value );
		static double getEpsilon( Eref e );
		static void setNeighDist( const Conn* c, double value );
		static double getNeighDist( Eref e );

	private:

		/**
		 * epsilon is the max deviation of surface-point from surface
		 * I think it refers to when the molecule is stuck to the surface.
		 * Need to check with Steven.
		 */
		double epsilon_;	

		/**
		 * neighdist is capture distance from one panel to another.
		 * When a molecule diffuses off one panel and is within neighdist
		 * of the other, it is captured by the second.
		 */
		double neighDist_; 
};

// Used by the Smoldyn solver
extern const Cinfo* initGeometryCinfo();

#endif // _Geometry_h

/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/


#ifndef _Surface_h
#define _Surface_h

/**
 * The Surface corresponds to the Smoldyn surfacestruct.
 * It manages multiple panels that define the surface.
 */
class Surface
{
	public:
		Surface();
		
		///////////////////////////////////////////////////
		// Field assignment functions
		///////////////////////////////////////////////////
		double getVolume() const;

		static const Cinfo* initCinfo();
	private:

		/**
		 * Volume is computed locally at reinit time. I can't really
		 * depend on the SmoldynHub to do this as surfaces may be
		 * independent of it. Basically this needs to query all the
		 * child panels and figure out from that. Non-trivial.
		 */
		double volume_; 
};

// Used by the Smoldyn solver
extern const Cinfo* initSurfaceCinfo();

#endif // _Surface_h

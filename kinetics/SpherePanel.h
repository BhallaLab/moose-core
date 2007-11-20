/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/


#ifndef _SpherePanel_h
#define _SpherePanel_h

/**
 * This defines a sphereangular panel. It is a subclass of the
 * Panel class which corresponds to the Smoldyn panelstruct.
 */
class SpherePanel: public Panel
{
	public:
		SpherePanel( unsigned int nDims = 3 );
		~SpherePanel() {
			;
		}
		
		///////////////////////////////////////////////////
		// Field assignment functions
		///////////////////////////////////////////////////

		unsigned int localNpts() const {
			return 2;
		}

		unsigned int localShapeId() const {
			return Moose::PSsph;
		}
		
		// Derived from Panel. Here we just fill in the spherical surface.
		void localFiniteElementVertices( 
			vector< double >& ret,  double area ) const;

	private:
};

#endif // _SpherePanel_h

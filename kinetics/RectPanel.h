/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/


#ifndef _RectPanel_h
#define _RectPanel_h

/**
 * This defines a rectangular panel. It is a subclass of the
 * Panel class which corresponds to the Smoldyn panelstruct.
 */
class RectPanel: public Panel
{
	public:
		RectPanel( unsigned int nDims = 3 );
		~RectPanel() {
			;
		}
		
		///////////////////////////////////////////////////
		// Field assignment functions
		///////////////////////////////////////////////////

		unsigned int localNpts() const {
			return 4;
		}

		unsigned int localShapeId() const {
			return Moose::PSrect;
		}
		
		// Derived from Panel. Here we just fill in the rectangular surface.
		void localFiniteElementVertices( 
			vector< double >& ret,  double area ) const;

	private:
};

#endif // _RectPanel_h

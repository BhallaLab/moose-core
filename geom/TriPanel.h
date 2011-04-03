/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/


#ifndef _TriPanel_h
#define _TriPanel_h

/**
 * This defines a triangular panel. It is a subclass of the
 * Panel class which corresponds to the Smoldyn panelstruct.
 */
class TriPanel: public Panel
{
	public:
		TriPanel( unsigned int nDims = 3 );
		~TriPanel() {
			;
		}
		
		///////////////////////////////////////////////////
		// Field assignment functions
		///////////////////////////////////////////////////

		unsigned int localNpts() const {
			return 3;
		}

		unsigned int localShapeId() const {
			return Moose::PStri;
		}
		
		// Derived from Panel. Here we just fill in the triangular surface.
		void localFiniteElementVertices( 
			vector< double >& ret,  double area ) const;

		static const Cinfo* initCinfo();

	private:
};

#endif // _TriPanel_h

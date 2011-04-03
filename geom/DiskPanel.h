/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/


#ifndef _DiskPanel_h
#define _DiskPanel_h

/**
 * This defines a disk panel. It is a subclass of the
 * Panel class which corresponds to the Smoldyn panelstruct.
 */
class DiskPanel: public Panel
{
	public:
		DiskPanel( unsigned int nDims = 3 );
		~DiskPanel() {
			;
		}
		
		///////////////////////////////////////////////////
		// Field assignment functions
		///////////////////////////////////////////////////

		unsigned int localNpts() const {
			return 2;
		}

		unsigned int localShapeId() const {
			return Moose::PSdisk;
		}
		
		// Derived from Panel. Here we just fill in the diskangular surface.
		void localFiniteElementVertices( 
			vector< double >& ret,  double area ) const;

		static const Cinfo* initCinfo();
	private:
};

#endif // _DiskPanel_h

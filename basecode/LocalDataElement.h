/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2013 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#ifndef _LOCAL_DATA_ELEMENT_H
#define _LOCAL_DATA_ELEMENT_H

class SrcFinfo;
class FuncOrder;

/**
 * This is the class for handling the local portion of a data element
 * that is distributed over many nodes.
 * Does block-wise partitioning between nodes.
 */
class LocalDataElement: public DataElement
{
	public:
		/**
		 * This is the main constructor, used by Shell::innerCreate
		 * which makes most Elements. Also used to create base
		 * Elements to init the simulator in main.cpp.
		 * Id is the Id of the new Element
		 * Cinfo is the class
		 * name is its name
		 * numData is the number of data entries, defaults to a singleton.
		 */
		LocalDataElement( Id id, const Cinfo* c, const string& name,
			unsigned int numData = 1 )

		/**
		 * This constructor copies over the original n times. It is
		 * used for doing all copies, in Shell::innerCopyElements.
		 */
		LocalDataElement( Id id, const Element* orig, unsigned int n, bool toGlobal);

		/**
		 * Virtual Destructor
		 */
		~LocalDataElement();

		/** 
		 * Virtual copier. Makes a copy of self.
		 */
		Element* copyElement( Id newParent, Id newId, unsigned int n, 
			bool toGlobal ) const;

		/////////////////////////////////////////////////////////////////
		// Information access fields
		/////////////////////////////////////////////////////////////////

		/// Inherited virtual. Returns number of data entries over all nodes
		unsigned int numData() const;

		/// Inherited virtual. Returns node location of specified object
		unsigned int getNode( DataId dataId ) const;

		/// Inherited virtual. Reports if this is Global, which it isn't
		bool isGlobal() const {
			return false;
		}

		/////////////////////////////////////////////////////////////////
		// data access stuff
		/////////////////////////////////////////////////////////////////

		/**
		 * Inherited virtual.
		 * Looks up specified field data entry. On regular objects just
		 * returns the data entry specified by the rawIndex. 
		 * On FieldElements like synapses, does a second lookup on the
		 * field index.
		 * Note that the index is NOT a
		 * DataId: it is instead the raw index of the data on the current
		 * node. Index is also NOT the character offset, but the index
		 * to the data array in whatever type the data may be.
		 *
		 * The DataId has to be filtered through the nodeMap to
		 * find a) if the entry is here, and b) what its raw index is.
		 *
		 * Returns 0 if either index is out of range.
		 */
		char* data( unsigned int rawIndex, 
						unsigned int fieldIndex = 0 ) const;

		/**
		 * Inherited virtual.
		 * Changes the total number of data entries on Element in entire
		 * simulation. Not permitted for
		 * FieldElements since they are just fields on the data.
		 */
		void resize( unsigned int newNumData );

		/////////////////////////////////////////////////////////////////

	private:
		/**
		 * This is the total number of data entries on this Element, in
		 * the entire simulation. Note that these 
		 * entries do not have to be on this node, some may be farmed out
		 * to other nodes.
		 */
		unsigned int numData_;

		/**
		 * This is the number of data entries per node, except for possibly
		 * the last node if they don't divide evenly. Useful for
		 * intermediate calculations.
		 */
		unsigned int numPerNode_;

};

#endif // _LOCAL_DATA_ELEMENT_H

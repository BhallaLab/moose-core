/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _NEUTRAL_H
#define _NEUTRAL_H

class Neutral
{
	public:
		friend istream& operator >>( istream& s, Neutral& d );
		friend ostream& operator <<( ostream& s, const Neutral& d );
		Neutral();

		/////////////////////////////////////////////////////////////////
		// Field access functions
		/////////////////////////////////////////////////////////////////

		/**
		 * Field access functions for the entire object. For Neutrals 
		 * the setThis function is a dummy: it doesn't do anything because
		 * the Neutral has no data to set. However, the function name acts
		 * as a placeholder and derived objects can override the function
		 * so that the entire object can be accessed as a field and also
		 * for inter-node data transfer.
		 */
		void setThis( Neutral v );

		/**
		 * Field access functions for the entire object. For Neutrals 
		 * the getThis function does return the Neutral object, but it
		 * has no data to set. However, the function name acts
		 * as a placeholder and derived objects can override the function
		 * so that the entire object can be accessed as a field and also
		 * used for inter-node data transfer.
		 */
		Neutral getThis() const;

		/**
		 * Field access functions for the name of the Element/Neutral
		 */
		void setName( const Eref& e, string name );
		string getName( const Eref& e ) const;

		/**
		 * Field access functions for the group of the Element
		 */
		void setGroup( const Eref& e, unsigned int val );
		unsigned int getGroup( const Eref& e ) const;

		/**
		 * Readonly field access function for getting all outgoing Msgs.
		 */
		vector< ObjId > getOutgoingMsgs( const Eref& e ) const;

		/**
		 * Readonly field access function for getting all incoming Msgs.
		 */
		vector< ObjId > getIncomingMsgs( const Eref& e ) const;

		/**
		 * Readonly field access function for getting Ids connected to
		 * current Id via specified Field.
		 * Field is specified by its name.
		 * Returns an empty vector if it fails.
		 */
		vector< Id > getNeighbours( const Eref& e, string field ) const;


		/**
		 * Simply returns own ObjId
		 */
		ObjId getObjId( const Eref& e ) const;

		/**
		 * Looks up the full Id info for the parent of the current Element
		 */
		ObjId getParent( const Eref& e ) const;

		/**
		 * Looks up all the Element children of the current Element
		 */
		vector< Id > getChildren( const Eref& e ) const;

		/**
		 * Builds a vector of all descendants of e
		 */
		unsigned int buildTree( const Eref& e, vector< Id >& tree ) const;

		/**
		 * Traverses to root, building path.
		 */
		string getPath( const Eref& e ) const;

		/**
		 * Looks up the Class name of the current Element
		 */
		string getClass( const Eref& e ) const;

		/**
		 * linearSize is the # of entries on Element. Its value is
		 * the product of all dimensions.
		 * Note that on a FieldElement this includes field entries.
		 * If field entries form a ragged array, then the linearSize may be
		 * greater than the actual number of allocated entries, since the
		 * fieldDimension is at least as big as the largest ragged array.
		 */
		unsigned int getLinearSize( const Eref& e ) const;

		/**
		 * Dimensions of data on the Element.
		 * This includes the fieldDimension if present.
		 */
		vector< unsigned int > getDimensions( const Eref& e ) const;

		/**
		 * Access function for the last (fastest varying) Dimension of the 
		 * data handler for the Element. For FieldDataHandlers this sets
		 * the max size of the ragged array for fields, such as synapses.
		 * For regular data handlers this sets the last dimension. It does
		 * permit you to scale from zero to N on the last dimension, and
		 * vice versa. A feeble attempt is made to retain existing data, but
		 * should not be counted on. Node balancing is done in accordance
		 * with whatever the last policy was.
		 * Note that this operation invalidates all DataIds and ObjIds
		 * that were set up for this Element. In the dubious event of your
		 * using iterators on the Element or its contents, those will be
		 * invalidated too.  
		 * Messages should remain intact.
		 */
		void setLastDimension( const Eref& e, unsigned int val );
		/**
		 * Access function for the last (fastest varying) Dimension of the 
		 * data handler for the Element. For FieldDataHandlers this gets
		 * the max size of the ragged array for fields, such as synapses.
		 * For regular data handlers this gets the last dimension.
		 */
		unsigned int getLastDimension( const Eref& e ) const;

		/** 
		 * Returns the vector of path index vectors for each dimension of
		 * the current object. Note that this is not the dimensions of these
		 * vectors, but the actual indices used to look up the object.
		 */
		vector< vector< unsigned int > > getPathIndices( 
			const Eref& e ) const;

		/**
		 * Gets the number of entries of a FieldElement on current node.
		 * If it is a regular Element, returns zero.
		 */
		unsigned int getLocalNumField( const Eref& e ) const;

		////////////////////////////////////////////////////////////
		// DestFinfo functions
		////////////////////////////////////////////////////////////

		/**
		 * Destroys Element and all children
		 */
		void destroy( const Eref& e, int stage );

		/**
		 * Request conversion of data into a blockDataHandler subclass,
		 * and to carry out node balancing of data as per args.
		 */
		void blockNodeBalance( const Eref& e, 
			unsigned int, unsigned int, unsigned int );

		/**
		 * Request conversion of data into a generalDataHandler subclass,
		 * and to carry out node balancing of data as per args.
		 */
		void generalNodeBalance( const Eref& e,
			unsigned int myNode, vector< unsigned int > nodeAssignment );
		

		////////////////////////////////////////////////////////////
		// Static utility functions
		////////////////////////////////////////////////////////////

		/**
		 * Finds specific named child
		 */
		static Id child( const Eref& e, const string& name );

		/**
		 * Returns parent object
		 */
		static ObjId parent( const Eref& e );
		static Id parent( Id id );

		/**
		 * Checks if 'me' is a descendant of 'ancestor'
		 */
		static bool isDescendant( Id me, Id ancestor );

		/**
		 * Standard initialization function, used whenever we want to
		 * look up the class Cinfo
		 */
		static const Cinfo* initCinfo();

		/**
		 * return ids of all the children in ret.
		 */
		static void children( const Eref& e, vector< Id >& ret );

		/**
		 * Finds the path of element e
		 */
		static string path( const Eref& e );

		/**
		 * Checks if specified field is a global, typically because it is
		 * present on the Element and therefore should be assigned uniformly
		 * on all nodes
		 */
		static bool isGlobalField( const string& field );

	private:
		// string name_;
};

#endif // _NEUTRAL_H

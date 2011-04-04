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
		void setName( const Eref& e, const Qinfo* q, string name );
		string getName( const Eref& e, const Qinfo* q ) const;

		/**
		 * Field access functions for the group of the Element
		 */
		void setGroup( const Eref& e, const Qinfo* q, unsigned int val );
		unsigned int getGroup( const Eref& e, const Qinfo* q ) const;

		/**
		 * Readonly field access function for getting all outgoing Msgs.
		 */
		vector< ObjId > getOutgoingMsgs(
			const Eref& e, const Qinfo* q ) const;

		/**
		 * Readonly field access function for getting all incoming Msgs.
		 */
		vector< ObjId > getIncomingMsgs(
			const Eref& e, const Qinfo* q ) const;

		/**
		 * Readonly field access function for getting source Ids
		 * that sent a Msg to the current Id.
		 * Field is specified by its name.
		 * Returns an empty vector if it fails.
		 */
		vector< Id > getMsgTargetIds( 
			const Eref& e, const Qinfo* q, string field ) const;

		/**
		 * Readonly field access function for getting destination Ids
		 * that receive Msgs from the current Id.
		 * Field is specified by its name.
		 * Returns an empty vector if it fails.
		 */
		vector< Id > getMsgSourceIds(
			const Eref& e, const Qinfo* q, string field ) const;


		/**
		 * Simply returns own ObjId
		 */
		ObjId getObjId( const Eref& e, const Qinfo* q ) const;

		/**
		 * Looks up the full Id info for the parent of the current Element
		 */
		ObjId getParent( const Eref& e, const Qinfo* q ) const;

		/**
		 * Looks up all the Element children of the current Element
		 */
		vector< Id > getChildren( const Eref& e, const Qinfo* q ) const;

		/**
		 * Builds a vector of all descendants of e
		 */
		unsigned int buildTree( const Eref& e, const Qinfo* q, 
			vector< Id >& tree ) const;

		/**
		 * Traverses to root, building path.
		 */
		string getPath( const Eref& e, const Qinfo* q ) const;

		/**
		 * Looks up the Class name of the current Element
		 */
		string getClass( const Eref& e, const Qinfo* q ) const;

		/**
		 * linearSize is the # of entries on Element. Its value is
		 * the product of all dimensions.
		 * Note that on a FieldElement this includes field entries.
		 * If field entries form a ragged array, then the linearSize may be
		 * greater than the actual number of allocated entries, since the
		 * fieldDimension is at least as big as the largest ragged array.
		 */
		unsigned int getLinearSize( const Eref& e, const Qinfo* q ) const;

		/**
		 * Dimensions of data on the Element.
		 * This includes the fieldDimension if present.
		 */
		vector< unsigned int > getDimensions( const Eref& e, const Qinfo* q ) const;

		/**
		 * Access function for the fieldDimension of the data handler
		 * for the Element. Ignored for objects that are not Fields.
		 */
		void setFieldDimension( const Eref& e, const Qinfo* q, unsigned int val );
		unsigned int getFieldDimension( const Eref& e, const Qinfo* q ) const;

		/**
		 * Destroys Element and all children
		 */
		void destroy( const Eref& e, const Qinfo* q, int stage );
		
		////////////////////////////////////////////////////////////

		/**
		 * Finds specific named child
		 */
		static Id child( const Eref& e, const string& name );

		/**
		 * Returns parent object
		 */
		static ObjId parent( const Eref& e );

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

	private:
		// string name_;
};

#endif // _NEUTRAL_H

/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _NEUTRAL_H
#define _NEUTRAL_H

/**
 * This class just supports the most basic object hierarchy stuff.
 * It can be both a parent and a child of objects.
 * Note that this is a data field, and is meant to sit on
 * a SimpleElement or some such.
 */
class Neutral
{
	public:
			enum DeleteStage { MARK_FOR_DELETION, CLEAR_MESSAGES,
					COMPLETE_DELETION };
			Neutral( )
			{;}

			static void childFunc( const Conn* c, int stage );
			static const string getName( Eref e );
			static const int getIndex( Eref e );
			static const int getId( Eref e );
			static const string getClass( Eref e );
			static unsigned int getNode( Eref e );

			/**
			 * getParent is a static utility function to return the
			 * parent of an element. Should really be on SimpleElement
			 * and should be a regular function.
			 */
			static Id getParent( Eref e );

			/**
			 * Returns a vector of child ids for this element
			 */
			static vector< Id > getChildList( Eref e );
			/**
			 * Gets list of children, but puts into a supplied vector.
			 * This is more efficient than getChildList.
			 * Mostly used in wildcarding.
			 * Note that the Ids of the children do not have node info
			 * internally. So we will have to convert if we want to
			 * send this off-node
			 */
			static void getChildren( const Eref e, vector< Id >& kids);

			static double getCpu( Eref e );
			static unsigned int getDataMem( Eref e );
			static unsigned int getMsgMem( Eref e );
			// static unsigned int getNode( const Element* e );
			// static void setNode( const Conn* c, unsigned int node );

			// The m in the name is to avoid confusion with the utility
			// function create below.
			static void mcreate( const Conn*,
							const string cinfo, const string name );
			static void mcreateArray( const Conn*,
							const string cinfo, const string name, int n );
			
			/**
			 * This version of the create function is meant for most use.
			 * Here we explicitly set the id of the new object
			 */
			static Element* create(
				const string& cinfo, const string& name, 
				Id parent, Id id );
			/**
			 * This version of create uses Id::scratchId() for the id of
			 * the new object
			static Element* create(
				const string& cinfo, const string& name, Element* parent );
			 */
			static Element* createArray(
				const string& cinfo, const string& name, 
				Id parent, Id id, int n );
			static void destroy( const Conn* c );
			static void setName( const Conn*, const string s );
			static void lookupChild( const Conn*, const string s );

			/**
 			* Looks up the child with the specified name, and returns its
			* id. For now don't deal with indices.
 			*/
			static Id getChildByName( Eref e, const string& s );

			// static const unsigned int childIndex;
			static vector< string > getFieldList( Eref elm );
};

#endif // _NEUTRAL_H

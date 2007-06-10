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

			static void childFunc( const Conn& c, int stage );
			static const string getName( const Element* e );
			static const string getClass( const Element* e );
			static unsigned int getParent( const Element* e );
			static vector< unsigned int > getChildList(
							const Element* e );
			static double getCpu( const Element* e );
			static unsigned int getDataMem( const Element* e );
			static unsigned int getMsgMem( const Element* e );
			// static unsigned int getNode( const Element* e );
			// static void setNode( const Conn& c, unsigned int node );

			// The m in the name is to avoid confusion with the utility
			// function create below.
			static void mcreate( const Conn&,
							const string cinfo, const string name );
			static Element* create(
				const string& cinfo, const string& name, 
				Element* parent );
			static void destroy( const Conn& c );
			static void setName( const Conn&, const string s );
			static void lookupChild( const Conn&, const string s );
			static unsigned int getChildByName( 
							const Element* e, const string& s );

			static const unsigned int childSrcIndex;
			static const unsigned int childIndex;
			static vector< string > getFieldList( const Element* elm );
};

#endif // _NEUTRAL_H

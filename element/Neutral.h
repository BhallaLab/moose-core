/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

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
			static unsigned int getParent( const Element* e );
			static void create( const Conn&,
							const string cinfo, const string name );
			static void destroy( const Conn& c );
			static void setName( const Conn&, const string s );
			static void lookupChild( const Conn&, const string s );
};

/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _EREF_H
#define _EREF_H

/**
 * Wrapper for Element and its index, for passing around as a unit.
 * Might as well use a pair, except that this is a more terse notation
 *
 * Don't go overboard on using this. It is meant for dealing with
 * sets, gets and sends. Most other things need to deal separately
 * with Element and index, or should use an Id.
 */
class Eref {
	public:
		Eref()
		{;}
		
		Eref( Element* eArg, unsigned int iArg = 0 )
			: e( eArg ), i( iArg )
		{;}
		
		void* data();

		bool operator<( const Eref& other ) const;

		Id id();
		
		Element* e;
		unsigned int i;
};

#endif // _EREF_H

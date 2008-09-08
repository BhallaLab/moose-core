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
 * Provides several utility functions including for messaging.
 *
 */

class ConnTainer; // required forward declaration.

class Eref {
	public:
		Eref()
		{;}
		
		Eref( Element* eArg, unsigned int iArg = 0 )
			: e( eArg ), i( iArg )
		{;}
		
		void* data();

		bool operator<( const Eref& other ) const;

		bool operator==( const Eref& other ) const;

		Element* operator->() {
			return e;
		}

		Id id();

		/**
		 * Returns the Element name with optional index if it is an
		 * array element.
		 */
		string name() const;
		
		/**
		 * Returns the Element name with index only if the parent element
		 * is simple and current element is array. Otherwise, returns the 
		 * name of the element. Faking business!!
		 */	
		string saneName(Id parent) const;

		///////////////////////////////////////////////////////////////
		// Msg handling functions
		///////////////////////////////////////////////////////////////

		/**
		 * Add a message from field f1 on current Element to field f2 on e2
		 * Return true if success.
		 */
		bool add( const string& f1, Eref e2, const string& f2,
			unsigned int connTainerOption );
		bool add( const string& f1, Eref e2, const string& f2 ); 
		// using default option

		/**
		 * Add a message from Msg m1 on current Element to Msg m2 on e2
		 * Return true if success.
		 */
		bool add( int m1, Eref e2, int m2, unsigned int connTainerOption );

		/**
		 * Drop slot 'doomed' on Msg msg
		 */
		bool drop( int msg, unsigned int doomed );

		/**
		 * Drop ConnTainer 'doomed' on Msg msg
		 */
		bool drop( int msg, const ConnTainer* doomed );

		/**
		 * Drop all msgs going out of the identified msg.
		 */
		bool dropAll( int msg );

		/**
		 * Drop all msgs going out of the identified Finfo.
		 */
		bool dropAll( const string& finfo );

		/**
		 * Drop all entries on a vector of connTainers. In due course
		 * this will be updated to be more efficient than just a sequence
		 * of individual calls to drop.
		 */
		bool dropVec( int msg, const vector< const ConnTainer* >& vec );

		/**
		 * Checks if the msgNum is OK. Looks at #finfos and #src.
		 * Rejects negative below #src, rejects others out of range.
		 * Does not consider indices into 'next' as valid.
		 */
		bool validMsg( int msg ) const;

		static Eref root();
		
		Element* e;
		unsigned int i;
};

#endif // _EREF_H

/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#ifndef _FINFO_H
#define _FINFO_H

// typedef unsigned int (*OpFunc )( Eref e, const void* buf );

class Finfo
{
	public:
		Finfo( const string& name, const string& doc );
		virtual ~Finfo() {;}

		const string& name() const;

		/**
		 * Make an identical copy of self. Used for derivation of classes.
		virtual Finfo* clone() const = 0;
		 */

		/**
		 * Assign function Ids, bindIndex and so on.
		 */
		virtual void registerFinfo( Cinfo* c ) = 0;

		/**
		 * Function to set this field using a string argument.
		 * Returns true on success.
		 * Normally called only from SetGet::strSet.
		 */
		virtual bool strSet( const Eref& tgt, const string& field,
			const string& arg ) const = 0;

		/**
		 * Function to return value of field into a string argument.
		 * Returns true on success.
		 * Normally called only from SetGet::strGet.
		 */
		virtual bool strGet( const Eref& tgt, const string& field,
			string& returnValue ) const = 0;

		/**
		 * This function is called on each new Element after
		 * it is created, in case any stuff needs doing. Typical
		 * uses are to create FieldElements automatically
		 * as soon as the parent is made.
		 * Used in FieldElementFinfo.h
		 */
		virtual void postCreationFunc( Id newId, Element* newElm ) const {
			;
		}

		/**
		 * Registers the Msg slot to be used for transmitting messages.
		 * SrcFinfos take the 'current' value and increment it, other
		 * msgs leave it alone.
		virtual BindIndex registerBindIndex( BindIndex current ) = 0;
		 */

		/**
		 * Checks that the type of target Finfo matches self, and is safe
		 * to exchange messages with.
		 * Is called only from msg src, so most Finfos return 0.
		 * SrcFinfo and SharedFinfo will need to implement this.
		 */
		virtual bool checkTarget( const Finfo* target) const {
			return 0;
		}

		/**
		 * Sets up specified Msg between src and dest. 
		 * Does all the type checking.
		 * Returns 1 on success.
		 * Defaults to 0 because many Finfo combinations won't work.
		 */
		virtual bool addMsg( const Finfo* target, MsgId mid, Element* src ) const
		{ 
			return 0;
		}

		////////////////////////////////////////////////////////////////
		// Functions for handling MOOSE Element inspection
		////////////////////////////////////////////////////////////////
		static const Cinfo* initCinfo();

	private:
		string name_;
		string doc_;
};

#endif // _FINFO_H

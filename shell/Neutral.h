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

class Neutral: public Data
{
	public:
		Neutral();
		void process( const ProcInfo* p, const Eref& e );

		/**
		 * Field access functions for the name of the Element/Neutral
		 */
		void setName( Eref e, const Qinfo* q, string name );
		string getName( Eref e, const Qinfo* q ) const;

		/**
		 * Simply returns own fullId
		 */
		FullId getFullId( Eref e, const Qinfo* q ) const;

		/**
		 * Looks up the full Id info for the parent of the current Element
		 */
		FullId getParent( Eref e, const Qinfo* q ) const;

		/**
		 * Looks up all the Element children of the current Element
		 */
		vector< Id > getChildren( Eref e, const Qinfo* q ) const;

		/**
		 * Traverses to root, building path.
		 */
		string getPath( Eref e, const Qinfo* q ) const;

		/**
		 * Looks up the Class name of the current Element
		 */
		string getClass( Eref e, const Qinfo* q ) const;

		/**
		 * Destroys Element and all children
		 */
		void destroy( Eref e, const Qinfo* q, int stage );

		/**
		 * Standard initialization function, used whenever we want to
		 * look up the class Cinfo
		 */
		static const Cinfo* initCinfo();

	private:
		// string name_;
};

#endif // _NEUTRAL_H

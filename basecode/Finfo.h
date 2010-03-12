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

		virtual void registerOpFuncs( 
			map< string, FuncId >& fm, vector< OpFunc* >& funcs ) = 0; 

		/**
		 * Registers the Msg slot to be used for transmitting messages.
		 * SrcFinfos take the 'current' value and increment it, other
		 * msgs leave it alone.
		 */
		virtual BindIndex registerBindIndex( BindIndex current ) = 0;

		/**
		 * Checks that the type of target Finfo matches self, and is safe
		 * to exchange messages with.
		 * Is called only from msg src, so most Finfos return 0.
		 * SrcFinfo and SharedFinfo will need to implement this.
		 */
		virtual bool checkTarget( const Finfo* target) const {
			return 0;
		}
		
	private:
		string name_;
		string doc_;
};

#endif // _FINFO_H

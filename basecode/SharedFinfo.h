/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#ifndef _SHARED_FINFO_H
#define _SHARED_FINFO_H

/**
 * This is a SharedFinfo, which wraps an arbitrary set of regular
 * Src and Dest Messages. Its main job is to do typechecking for setting
 * up multiple data streams to go across the same Msg. 
 */

class SharedFinfo: public Finfo
{
	public:
		SharedFinfo( const string& name, const string& doc, 
			Finfo** entries, unsigned int numEntries );

		~SharedFinfo() {;}

		void registerOpFuncs(
			map< string, FuncId >& fnames, vector< OpFunc* >& funcs );

		BindIndex registerBindIndex( BindIndex current );

		/**
		 * Checks that the type of target Finfo matches self, and is safe
		 * to exchange messages with.
		 */
		bool checkTarget( const Finfo* target ) const;

		/**
		 * First calls checkTarget on all targets, then sets up message.
		 * Returns true on success.
		 */
		bool addMsg( const Finfo* target, MsgId mid, Element* src ) const;

	private:
		vector< const SrcFinfo* > src_;
		vector< Finfo* > dest_;
};

#endif // _SHARED_FINFO_H

/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
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
		~Finfo() {;}

		const string& name() const;

		virtual void registerOpFuncs( 
			map< string, FuncId >& fm, vector< OpFunc* >& funcs ) = 0; 
		
	private:
		string name_;
		string doc_;
};


/*
 * Should set this up to take a regular func, similar to field funcs.
 * Will need a bit of templating but will be easier for users.
class DestFinfo: public Finfo
{
	public:
		DestFinfo( const string& name, OpFunc func, const string& doc );
		void registerOpFuncs( 
			map< string, FuncId >& fnames, vector< Ftype* >& funcs );

	private:
		OpFunc func_;
};
*/

#endif // _FINFO_H

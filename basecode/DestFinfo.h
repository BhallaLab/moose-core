/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#ifndef _DEST_FINFO_H
#define _DEST_FINFO_H


/*
 * DestFinfo manages function requests. All messages terminate on
 * the functions stored in DestFinfos. No templates here, as all the
 * type-specific work is done by the OpFunc.
*/
class DestFinfo: public Finfo
{
	public:
		~DestFinfo();
		DestFinfo( const string& name, const string& doc,OpFunc* func );
		void registerOpFuncs( 
			map< string, FuncId >& fnames, vector< OpFunc* >& funcs );
		BindIndex registerBindIndex( BindIndex current );
		const OpFunc* getOpFunc() const;
		FuncId getFid() const;

	private:
		OpFunc* func_;
		FuncId fid_;
};

#endif // _DEST_FINFO_H

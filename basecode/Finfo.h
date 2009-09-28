/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

typedef unsigned int (*OpFunc )( Eref e, const void* buf );

class Finfo
{
	public:
		Finfo( OpFunc op, const string& name );
		~Finfo() {;}

		unsigned int op( Eref e, const void* buf ) const;

		const string& name() const;

		virtual void registerOpFuncs( 
			map< OpFunc, FuncId >& fm, vector< OpFunc >& funcs ); 
		
	private:
		OpFunc op_;
		string name_;
};

/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"

DestFinfo::~DestFinfo() {
	delete func_;
}

DestFinfo::DestFinfo( const string& name, const string& doc, 
	OpFunc* func )
	: Finfo( name, doc ), func_( func )
{
	;
}

void DestFinfo::registerFinfo( Cinfo* c )
{
	fid_ = c->registerOpFunc( func_ );
//	cout << c->name() << "." << name() << ": " << fid_ << endl;
}

const OpFunc* DestFinfo::getOpFunc() const
{
	return func_;
}

FuncId DestFinfo::getFid() const
{
	return fid_;
}

// For now we bail. Later we can update OpFunc to dig up the correct SetGet
SetGet* DestFinfo::getSetGet( const Eref& e ) const
{
	return 0;
}

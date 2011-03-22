/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include <typeinfo>
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

bool DestFinfo::strSet( 
	const Eref& tgt, const string& field, const string& arg ) const 
{
	return func_->strSet( tgt, field, arg );
}

bool DestFinfo::strGet( 
	const Eref& tgt, const string& field, string& returnValue ) const 
{
	return 0;
}

string DestFinfo::rttiType() const
{
	return func_->rttiType();
}

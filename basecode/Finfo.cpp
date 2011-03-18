/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include <typeinfo>
#include "header.h"

Finfo::Finfo( const string& name, const string& doc )
	: name_( name ), doc_( doc )
{
	;
}

////////////////////////////////////////////////////////////////
// MOOSE class functions
////////////////////////////////////////////////////////////////

const Cinfo* Finfo::initCinfo()
{
	static ReadOnlyValueFinfo< Finfo, string > name(
		"name",
		"Name of Finfo",
		&Finfo::name
	);

	static ReadOnlyValueFinfo< Finfo, string > docs(
		"docs",
		"Documentation for Finfo",
		&Finfo::docs
	);

	static ReadOnlyValueFinfo< Finfo, string > type(
		"type",
		"RTTI type info for this Finfo",
		&Finfo::type
	);

	static ReadOnlyValueFinfo< Finfo, vector< string > > src(
		"src",
		"Subsidiary SrcFinfos. Useful for SharedFinfos",
		&Finfo::src
	);
	static ReadOnlyValueFinfo< Finfo, vector< string > > dest(
		"dest",
		"Subsidiary DestFinfos. Useful for SharedFinfos",
		&Finfo::dest
	);


	static Finfo* finfoFinfos[] = {
		&name,	// ReadOnlyValue
		&docs,		// ReadOnlyValue
		&type,		// ReadOnlyValue
		&src,		// ReadOnlyValue
		&dest,		// ReadOnlyValue
	};

	static Cinfo finfoCinfo (
		"Finfo",
		Neutral::initCinfo(),
		finfoFinfos,
		sizeof( finfoFinfos ) / sizeof( Finfo* ),
		new Dinfo< short > // dummy
	);

	return &finfoCinfo;
}

static const Cinfo* finfoCinfo = Finfo::initCinfo();

////////////////////////////////////////////////////////////////

string Finfo::name( ) const
{
	return name_;
}

string Finfo::docs( ) const
{
	return doc_;
}


string Finfo::type( ) const
{
	return this->rttiType();
}

vector< string > Finfo::src( ) const
{
	return this->innerSrc();
}

vector< string > Finfo::dest( ) const
{
	return this->innerDest();
}


// Default virtual functions.
string Finfo::rttiType() const
{
	return typeid( *this ).name();
}

vector< string > Finfo::innerSrc() const
{
	static vector< string > ret;
	return ret;
}

vector< string > Finfo::innerDest() const
{
	static vector< string > ret;
	return ret;
}

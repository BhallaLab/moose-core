/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "SetGet.h"
#include "Dinfo.h"

const Cinfo* SetGet::initCinfo()
{
	/*
	static Finfo* reacFinfos[] = {
		new Finfo( setKf_ ),
		new Finfo( setKb_ ),
	};
	*/
	static Finfo* setGetFinfos[] = {
		new ValueFinfo< SetGet, string >( 
			"name",
			"Name of object", 
			&SetGet::setName, 
			&SetGet::getName ),
	};

	static Cinfo setGetCinfo (
		"SetGet",
		0, // No base class.
		setGetFinfos,
		sizeof( setGetFinfos ) / sizeof( Finfo* ),
		new Dinfo< SetGet >()
	);

	return &setGetCinfo;
}

static const Cinfo* setGetCinfo = SetGet::initCinfo();


SetGet::SetGet()
	: name_( "" )
{
	;
}

void SetGet::process( const ProcInfo* p, Eref e )
{
	;
}

void SetGet::setName( const string& name )
{
	name_ = name;
}

const string& SetGet::getName() const
{
	return name_;
}

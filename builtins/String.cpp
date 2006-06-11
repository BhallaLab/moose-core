/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2005 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU General Public License version 2
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#include "header.h"
#include "String.h"

//////////////////////////////////////////////////////////////////
// Strings are Elements that can have children. They have no 
// data fields. They equivalent to the GENESIS neutrals.
//////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////
// String initialization
//////////////////////////////////////////////////////////////////

Finfo* String::fieldArray_[] = 
{
	new ValueFinfo< string >("value", 
		&String::get, &String::set, "string")
};

const Cinfo String::cinfo_(
	"String",
	"Upinder S. Bhalla, NCBS",
	"String class. Holds a string.",
	"Element",
	String::fieldArray_,
	sizeof(String::fieldArray_)/sizeof(Finfo *),
	&String::create
);


//////////////////////////////////////////////////////////////////
// String functions
//////////////////////////////////////////////////////////////////

String::~String()
{
	;
}

// In this function we take the proto's string value and its name.
Element* String::create(
			const string& name, Element* parent, const Element* proto)
{
	String* ret = new String(name);
	const String* p = dynamic_cast<const String *>(proto);
	if (p)
		ret->value_ = p->value_;
	return ret;
	/*
	if (parent->adoptChild(ret)) {
		return ret;
	} else {
		delete ret;
		return 0;
	}
	return 0;
	*/
}

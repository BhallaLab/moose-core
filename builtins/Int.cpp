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
#include "Int.h"

//////////////////////////////////////////////////////////////////
// Ints are Elements that can have children. They have no 
// data fields. They equivalent to the GENESIS neutrals.
//////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////
// Int initialization
//////////////////////////////////////////////////////////////////

Finfo* Int::fieldArray_[] = 
{
	new ValueFinfo< int >("value", &Int::get, &Int::set, "int"),
	new ValueFinfo< string >("strval", &Int::getstr, &Int::setstr,
		"string")
};

const Cinfo Int::cinfo_(
	"Int",
	"Upinder S. Bhalla, NCBS",
	"Int class. Holds an int.",
	"Element",
	Int::fieldArray_,
	sizeof(Int::fieldArray_)/sizeof(Finfo *),
	&Int::create
);


//////////////////////////////////////////////////////////////////
// Int functions
//////////////////////////////////////////////////////////////////

Int::~Int()
{
	;
}

// In this function we take the proto's int value and its name.
Element* Int::create(
			const string& name, Element* parent, const Element* proto)
{
	Int* ret = new Int(name);
	const Int* p = dynamic_cast<const Int *>(proto);
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

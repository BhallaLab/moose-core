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
#include "Double.h"

//////////////////////////////////////////////////////////////////
// Doubles are Elements that can have children. They have no 
// data fields. They equivalent to the GENESIS neutrals.
//////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////
// Double initialization
//////////////////////////////////////////////////////////////////

Finfo* Double::fieldArray_[] = 
{
	new ValueFinfo< double >("value", &Double::get, &Double::set, "double"),
	new ValueFinfo< string >("strval", &Double::getstr, &Double::setstr,
		"string")
};

const Cinfo Double::cinfo_(
	"Double",
	"Upinder S. Bhalla, NCBS",
	"Double class. Holds a double.",
	"Element",
	Double::fieldArray_,
	sizeof(Double::fieldArray_)/sizeof(Finfo *),
	&Double::create
);


//////////////////////////////////////////////////////////////////
// Double functions
//////////////////////////////////////////////////////////////////

Double::~Double()
{
	;
}

// In this function we take the proto's int value and its name.
Element* Double::create(
			const string& name, Element* parent, const Element* proto)
{
	Double* ret = new Double(name);
	const Double* p = dynamic_cast<const Double *>(proto);
	if (p)
		ret->value_ = p->value_;
	return ret;
}

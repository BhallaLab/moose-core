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
#include "TestField.h"

//////////////////////////////////////////////////////////////////
// This is a class for testing the automatic conversion of fields.
//////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////
// TestField initialization
//////////////////////////////////////////////////////////////////

Finfo* TestField::fieldArray_[] = 
{
	new ValueFinfo< int >(
		"i", &TestField::getI, &TestField::setI, "int"),
	new ValueFinfo< int >(
		"j", &TestField::getJ, &TestField::setJ, "int"),
	new ValueFinfo< string >(
		"s", &TestField::getS, &TestField::setS, "string"),
	new ArrayFinfo< int >(
		"v", &TestField::getV, &TestField::setV, "int")
};

const Cinfo TestField::cinfo_(
	"TestField",
	"Upinder S. Bhalla, NCBS",
	"TestField class. Holds some simple fields for testing assignment.",
	"Element",
	TestField::fieldArray_,
	sizeof(TestField::fieldArray_)/sizeof(Finfo *),
	&TestField::create
);


//////////////////////////////////////////////////////////////////
// TestField functions
//////////////////////////////////////////////////////////////////

TestField::~TestField()
{
	;
}

// In this function we take the proto's value
Element* TestField::create(
			const string& name, Element* parent, const Element* proto)
{
	TestField* ret = new TestField(name);
	const TestField* p = dynamic_cast<const TestField *>(proto);
	if (p) {
		ret->i_ = p->i_;
		ret->j_ = p->j_;
		ret->s_ = p->s_;
	}
	if (parent->adoptChild(ret)) {
		return ret;
	} else {
		delete ret;
		return 0;
	}
	return 0;
}

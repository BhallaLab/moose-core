/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "header.h"
#include "TableEntry.h"
#include "Table.h"
// #include "UpFunc.h"

const Cinfo* TableEntry::initCinfo()
{
		static ValueFinfo< TableEntry, double > value(
			"value",
			"Data value in this entry",
			&TableEntry::setValue,
			&TableEntry::getValue
		);

	static Finfo* tableEntryFinfos[] = {
		&value,	// Field
	};

	static Cinfo tableEntryCinfo (
		"TableEntry",
		Neutral::initCinfo(),
		tableEntryFinfos,
		sizeof( tableEntryFinfos ) / sizeof ( Finfo* ),
		new Dinfo< TableEntry >()
	);

	return &tableEntryCinfo;
}

static const Cinfo* tableEntryCinfo = TableEntry::initCinfo();

TableEntry::TableEntry()
	: value_( 1.0 )
{
	;
}

TableEntry::TableEntry( double v )
	: value_( v )
{
	;
}

void TableEntry::setValue( const double v )
{
	value_ = v;
}

double TableEntry::getValue() const
{
	return value_;
}

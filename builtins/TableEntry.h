/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _TABLE_ENTRY_H
#define _TABLE_ENTRY_H

/**
 * Just a double in the Table array.
 */
class TableEntry
{
	public: 
		TableEntry();
		TableEntry( double v );

		void setValue( const double v );
		double getValue() const;

		static const Cinfo* initCinfo();
	private:
		/**
		 * TableEntry condition. Reflective = 1. Completely diffusive = 0
		 * Unless it is completely reflective, there should be an adjacent
		 * compartment into which the molecules diffuse.
		 */
		double value_;
};

#endif	// _TABLE_ENTRY_H

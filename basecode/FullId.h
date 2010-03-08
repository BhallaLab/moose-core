
/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _FULL_ID_H
#define _FULL_ID_H

/**
 * This class manages lookups for specific data entries in elements,
 * in a node-independent manner.
 * It is basically a composite of Id and DataId.
 */
class FullId
{
	friend ostream& operator <<( ostream& s, const FullId& i );
	friend istream& operator >>( istream& s, FullId& i );
	public:
		//////////////////////////////////////////////////////////////
		//	FullId creation
		//////////////////////////////////////////////////////////////
		/**
		 * Returns the root Id
		 */
		FullId()
			: id(), dataId()
		{;}

		/**
		 * Creates a FullId using specified Id and DataId
		 */
		FullId( Id i, DataId d )
			: id( i ), dataId( d )
		{;}

		/**
		 * Here are the data values.
		 */
		Id id; 
		DataId dataId;

	private:
};

#endif // _FULL_ID_H

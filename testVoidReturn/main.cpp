/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

/**
 * This file is to test whether we could use the same template for handling
 * 'set' calls, which are voids, as we use for handling 'get' calls.
 * Seems like it can be done provided we specialize the void function
 * call.
 */


template< class T > T func()
{
	T foo;
	return foo;
}

template<> void func()
{
	return;
}

int main()
{
	func< int >();
	func< double >();
	func< bool >();
	func< void >();
}

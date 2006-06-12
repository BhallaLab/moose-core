/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _ForwardEuler_h
#define _ForwardEuler_h
class ForwardEuler
{
	friend class ForwardEulerWrapper;
	public:
		ForwardEuler()
		{
			isInitialized_ = 0;
		}

	private:
		int isInitialized_;
		vector< double >* y_;
		vector< double > yprime_;
};
#endif // _ForwardEuler_h

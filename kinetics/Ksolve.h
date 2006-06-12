/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/


#ifndef _Ksolve_h
#define _Ksolve_h
class Ksolve
{
	friend class KsolveWrapper;
	public:
		Ksolve()
		{
		}

	private:
		string path_;
};
#endif // _Ksolve_h

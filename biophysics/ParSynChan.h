/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
/**
 * This class is a parallel version of SynChan class.
 * This file is compiled only when the parallelizatin flag is enabled.
 * This class derives from SynChan. 
 *
 * 
 * This class refers to the base class, SynChan, for all functionality. 
 * Parallel moose parser would require overriding of some of the base class functionality. 
 * Such functions would be overridden in this class. 
 *
 * 
 */


#ifndef _ParSynChan_h
#define _ParSynChan_h

class ParSynChan : public SynChan
{
	public:
		ParSynChan();

};
#endif // _ParSynChan_h

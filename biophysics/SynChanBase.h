#ifndef _SynChanBase_h
#define _SynChanBase_h
/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2011 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
*********************************************************************
*/

/**
 * The SynChanBase combines the Synapse handling data and functions with 
 * the Channel handling stuff. It is therefore a dual-inherited
 * class. Nothing is added to this, it is used only as a base class
 * for various SynChans, all of which do quite different things.
 * MOOSE does not understand dual inheritance, so I implement the
 * MOOSE part of it by inheriting in the usual way from ChanBase,
 * and defining identical MOOSE fields for the SynBase part. This 
 * introduces the need to maintain identity between the fields for
 * SynBase and SynChanBase, which
 * will be a maintenance issue to keep on top of.
 */

class SynChanBase: public SynBase, public ChanBase
{
	public:
		SynChanBase();
		~SynChanBase();

	static const Cinfo* initCinfo();
};


#endif // _SynChanBase_h

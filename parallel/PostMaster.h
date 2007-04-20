/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _POST_MASTER_H
#define _POST_MASTER_H

/**
 * A skeleton class for starting out the postmaster.
 */
class PostMaster
{
	public:
		PostMaster();
		static unsigned int getMyNode( const Element* e );
		static unsigned int getRemoteNode( const Element* e );
		static void setRemoteNode( const Conn& c, unsigned int node );

	private:
		unsigned int localNode_;
		unsigned int remoteNode_;
};

#endif // _POST_MASTER_H

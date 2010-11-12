/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2010 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

class TickPtr
{
	public:
		TickPtr( TickMgr* mgr ) 
			: mgr_( mgr )
		{;}

		bool operator<( const TickPtr& other ) const {
			return ( mgr_->getNextTime() < other.mgr_->getNextTime() );
		}

		TickMgr* mgr() const {
			return mgr_;
		}

	private:
		TickMgr* mgr_;
};

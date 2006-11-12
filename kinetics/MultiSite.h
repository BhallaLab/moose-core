/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-2006 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#ifndef _MultiSite_h
#define _MultiSite_h
class MultiSite
{
	friend class MultiSiteWrapper;
	public:
		MultiSite()
			: nTotal_( 0.0), states_(1, 0), occupancy_(1, 1.0),
				rates_(1, 1.0), fraction_( 1, 0.0)
		{
		}

	private:
		double nTotal_;
		vector < int > states_;
		vector < double > occupancy_;
		vector < double > rates_;
		vector< double > fraction_;
};
#endif // _MultiSite_h

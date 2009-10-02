/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

/**
 * This class manages information going into and out of the async queue.
 */
class Qinfo
{
	public:
		Qinfo( FuncId f, unsigned int s, MsgId m )
			: f_( f ), size_( s ), m_( m )
		{;}
		Qinfo( const char* buf );

		void addToQ( vector< char >& q_, const char* arg ) const;

		FuncId fid() const {
			return f_;
		}
		MsgId mid() const {
			return m_;
		}
		unsigned int size() const {
			return size_;
		}
		
	private:
		FuncId f_;
		unsigned int size_; // size of argument in bytes.
		MsgId m_;
};

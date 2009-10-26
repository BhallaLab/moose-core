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
		Qinfo( FuncId f, unsigned int srcIndex, unsigned int size );

		Qinfo( FuncId f, unsigned int srcIndex, 
			unsigned int size, bool useSendTo );

		Qinfo( const char* buf );

		void setMsgId( MsgId m ) {
			m_ = m;
		}

		void addToQ( vector< char >& q_, const char* arg ) const;

		bool useSendTo() const {
			return useSendTo_;
		}

		MsgId mid() const {
			return m_;
		}

		FuncId fid() const {
			return f_;
		}

		unsigned int srcIndex() const {
			return srcIndex_;
		}

		unsigned int size() const {
			return size_;
		}

		/**
		 * Used when adding space for the index in sendTo
		 */
		void expandSize();

	private:
		MsgId m_;
		bool useSendTo_;
		FuncId f_;
		unsigned int srcIndex_; // Index of src.
		unsigned int size_; // size of argument in bytes.
};

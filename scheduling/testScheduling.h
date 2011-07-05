/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

class TestSched
{
	public:
		/** 
		 * This may be created as an array, but only on one thread
		 */
		TestSched()
			: index_( 0 )
		{
			if ( isInitPending_ ) {
				globalIndex_ = 0;
				pthread_mutex_init( &mutex_, NULL );
				isInitPending_ = 0;
			}
		}
		~TestSched()
		{
			if ( !isInitPending_ )
				pthread_mutex_destroy( &mutex_ );
			isInitPending_ = 1;
		}

		void process( const Eref& e, ProcPtr p );

		void zeroIndex() {
			index_ = 0 ;
		}

		static const Cinfo* initCinfo();
	private:
		int index_;
		static pthread_mutex_t mutex_;
		static int globalIndex_;
		static bool isInitPending_;
};


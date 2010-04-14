/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

static vector< unsigned int > _dims;

class testSchedElement: public Element
{
	public:
		testSchedElement() 
			: Element( Id::nextId(), Tick::initCinfo(), "testSched", _dims ),
			index_( 0 )
		{;}
		
		void process( const ProcInfo* p ) {
			static const int timings[] = { 1, 2, 2, 2, 3, 3, 4, 4, 4, 
				5, 5, 5, 6, 6, 6, 6, 7, 8, 8, 8, 9, 9, 10, 10, 10, 10, 10,
				11, 12, 12, 12, 12, 13, 14, 14, 14, 15, 15, 15, 15,
				16, 16, 16, 17, 18, 18, 18, 18, 19, 20, 20, 20, 20, 20 };
			unsigned int max = sizeof( timings ) / sizeof( int );
			// cout << "timing[ " << index_ << " ] = " << timings[ index_ ] << ", time = " << p->currTime << endl;
			assert( static_cast< int >( p->currTime ) == 	
				timings[ index_++ ] );
			assert( index_ <= max );
			// cout << index_ << ": " << p->currTime << endl;
		}
	private:
		unsigned int index_;
};

class testThreadSchedElement: public Element
{
	public:
		testThreadSchedElement() 
			// : Element( Tick::initCinfo(), 0, 0, 0, 0 ), index_( 0 )
			: Element( Id::nextId(), Tick::initCinfo(), 
				"testThreadSched", _dims ), index_( 0 )
		{ 
			pthread_mutex_init( &mutex_, NULL );
		}

		~testThreadSchedElement()
		{
			pthread_mutex_destroy( &mutex_ );
		}
		
		void process( const ProcInfo* p ) {
			static const int timings[] = { 1, 2, 2, 2, 3, 3, 4, 4, 4, 
				5, 5, 5, 6, 6, 6, 6, 7, 8, 8, 8, 9, 9, 10, 10, 10, 10, 10,
				11, 12, 12, 12, 12, 13, 14, 14, 14, 15, 15, 15, 15,
				16, 16, 16, 17, 18, 18, 18, 18, 19, 20, 20, 20, 20, 20 };
			unsigned int max = sizeof( timings ) / sizeof( int );
			// cout << "timing[ " << index_ << ", " << p->threadId << " ] = " << timings[ index_ / p->numThreads ] << ", time = " << p->currTime << endl;
			if ( static_cast< int >( p->currTime ) != 	
				timings[ index_ / p->numThreads ] )
				/*
				cout << "testThreadSchedElement::process: index= " << index_ << ", numThreads = " <<
					p->numThreads << ", currTime = " << p->currTime << 
					", mynode = " << p->nodeIndexInGroup << endl;
			*/
			assert( static_cast< int >( p->currTime ) == 	
				timings[ index_ / p->numThreadsInGroup ] );

			pthread_mutex_lock( &mutex_ );
				++index_;
			pthread_mutex_unlock( &mutex_ );

			assert( index_ <= max * p->numThreads );
			// cout << index_ << ": " << p->currTime << endl;
		}
	private:
		pthread_mutex_t mutex_;
		unsigned int index_;
};


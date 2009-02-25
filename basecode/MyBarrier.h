/**
 * An attempt to make a somewhat faster variant of the barrier, but
 * fast only if the system has put each thread on a separate processor.
 * It uses a busy loop to help things along.
 */
class MyBarrier 
{
	public:
		MyBarrier( unsigned int numThreads )
			: numThreads_( numThreads ), cycle_( 0 ), counter_( 0 )
		{
			assert( numThreads > 1 );
			threadCycle_ = new unsigned int[ numThreads ];
			for ( unsigned int i = 0; i < numThreads; ++i )
				threadCycle_[ i ] = 0;
			int status = pthread_mutex_init( &mutex_, 0 );
			if ( status )
				barf();
		}

		~MyBarrier() 
		{
			int status = pthread_mutex_destroy( &mutex_ );
			delete[] threadCycle_;
			if  ( status )
				barf();
		}

		void wait( unsigned int threadNum )
		{
			int status = pthread_mutex_lock( &mutex_ );
			if ( status ) {
				barf();
			}
				++counter_;
				if ( counter_ == numThreads_ ) {
					++cycle_;
					counter_ = 0;
				}
			pthread_mutex_unlock( &mutex_ );
			/*
			cout << "thread=" << threadNum << ", cycle = " << cycle_ <<
				", counter = " << counter_ << 
				", tc[0] = " << threadCycle_[0] <<
				", tc[1] = " << threadCycle_[1] <<
				endl << flush;
				*/

			// Busy loop. Expensive unless you have multicore machines.
			while( cycle_ == threadCycle_[ threadNum ] )
				;
			threadCycle_[ threadNum ]++;
		}
		void barf()
		{
			cout << "MyBarrier: Bad pthreads status\n";
			exit( 1 );
		}
	private:
		unsigned int numThreads_;
		volatile unsigned int cycle_;
		unsigned int counter_;
		unsigned int *threadCycle_;
		pthread_mutex_t mutex_;
};

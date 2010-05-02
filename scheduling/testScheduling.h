/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2009 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

class TestSched: public Data
{
	public:
		TestSched()
			: index_( 0 )
		{
			pthread_mutex_init( &mutex_, NULL );
		}
		~TestSched()
		{
			pthread_mutex_destroy( &mutex_ );
		}

		void process( const ProcInfo*p, const Eref& e );

		void eprocess( Eref e, const Qinfo* q, ProcPtr p ) {
			process( p, e );
		}

		static const Cinfo* initCinfo();
	private:
		pthread_mutex_t mutex_;
		unsigned int index_;
};


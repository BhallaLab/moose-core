template < class T > void send1( Eref e, Slot src, T val )
{
	double* buf = e.getBufPtr( src );
	*static_cast< T* >( buf ) = val;
}

template < class T1, class T2 > void send2( Eref e, Slot src, T1 v1, T2 v2 )
{
	double* buf = e.getBufPtr( src );
	*static_cast< T1* >( buf ) = v1;
	buf += sizeof( T1 ) / sizeof( double );
	*static_cast< T2* >( buf ) = v2;
}

// This is shifted over to a function of Eref
//void sendSpike( Eref e, Slot src, double t );

#if 0

/**
 * Send data into a queue. Need to add info about identity of sender.
 */
template < class T > void qSend1( Eref e, Slot src, T val )
{
	// TargetVec includes target Q info as well as weight, we just add
	// the T val.
	vector< TargetVec< T > > &v = e.getTargetVec( src );
	for( TargetVec::iterator i = v.begin(); i != v.end(); ++i )
		i->insertIntoQ( val );
}

void synSend( Eref e, Slot syn, double time ) {
	SynTargets *t = e.getSynMsg( syn );
	t->pushQ( time );
}

/**
 * Send an action potential through a projection Msg and insert it into
 * target-specific queues.
 */
void synSend( Eref e, Slot syn, double time ) {
	e.getSynTarget( syn, e.i )->pushQ( time ); 
	// Element index specifies which tgt
}

/**
 * SynTargets is the input-specific list of targets and associated data.
 * There are many SynTarget data structures in the Msg? This may be a bit
 * problematical. But it saves on lookups for the sparse matrix version.
 */
class SynTargets
{
	public:
		virtual void pushQ( double time ) = 0;
	private:
};

/**
 * Queue for synaptic events.
 */
class SynQ
{
};
#endif

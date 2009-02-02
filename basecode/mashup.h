/**
 * Carries all info needed by recvfunc to figure out what to do and which
 * object to do it on.
 * Perhaps the data field will no longer be needed.
 */
class Conn
{
	public:
		Eref target();
		Eref source();
		ConnRef targetInfo();
		ConnRef srcInfo();
		void* data() const;
};

/**
 * Specifies the projection.
 * Pure virtual base class, since many kinds of conntainers use this
 * interface.
 * Design decision: Permit multiple src/target Elements or only one?
 * If multiple, would presumably be built on a lower-level form.
 */
class ConnTainer
{
	public:
		Id src;
		Id dest;
		vector< >& srcVec( );
};


template< class T > Send( Eref& e, Msg m, T& arg );
{
	ConnTainer* ct = e.projection( );
	Projection* p = e.projection( );
	Connection* p = e.connection( m );
	p->setTargetBuf( m, static_cast< void* >( &arg ), sizeof( T ) );
	// specializations needed for arrays, strings etc.
	p->setTargetBuf( m, Ftype< T >::argConvert( arg1, arg2 ) );
	p->setQueue( m );
};


/**
 * Specifies either a single Element or a complete array.
 * If we're going to specify an array, should we extend to specify any
 * subset on a given Element?
 */
class Eref
{
	public:
		Element* e;
		IndexRange* i;
		void* data_;
		Index
		unsigned int numEntries_;

};

class IndexRange
{
	public:
		unsigned int total();
		void fullList ( vector< unsigned int >& ) const;
		vector< unsigned int >::iterator begin();
		vector< unsigned int >::iterator end();
		void foreach( ( void )( *func )( unsigned int  ) );
};

/**
 * Variants:
 * SingleIndex
 * AllIndices
 * ContigRange
 * Combo (MultipleContigs)
 * Sparse (Multiple individual indices)
 */

/**
 * Have a single Element class, that uses an IndexRange. There can
 * be multiple such objects representing a complete array. So the Id
 * must know how to access them all. Or they each refer to each other.
 * Or the user only sees the master (but may be many if distrib over nodes)
 */
class Element
{
	public:
	private:
		vector< MsgSrc > m_;
		vector< MsgBuf > b_;
		Data* d_;
};

class Data
{
	public:
		virtual void process( const ProcInfo *p ) = 0;
		virtual void reinit() = 0;
		virtual void event( unsigned int eventId ) = 0;
};

class MsgSrc
{
	public:
	private:
		ConnTainer* c_; // Holds the projection pattern.
		// Somewhere here I have to specify dest funcs. The connTainer
		// must not do so, because it serves many msgs, and is
		// used bidirectionally.
		// Could have vec of connTainers with matching funcs.
		// If ConnTainers can be nested, this will be general.
		MsgId id_; // Unique identifier for msg, includes type info.
		// Can't fold dest func into this. Consider reacs, which handle
		// identical args for sub and prd funcs, both from molecules.
};

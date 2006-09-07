#ifndef _TockWrapper_h
#define _TockWrapper_h
class TockWrapper: 
	public Tock, public Neutral
{
	friend Element* tickConnLookup( const Conn* );
    public:
		TockWrapper(const string& n)
		:
			Tock( n ), Neutral( n )
			// tickConn uses a templated lookup function
		{
			;
		}
///////////////////////////////////////////////////////
//    Field header definitions.                      //
///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
// Msgsrc header definitions .                       //
///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
// dest header definitions .                         //
///////////////////////////////////////////////////////
		static void processFunc( Conn* c, ProcInfo info ) {
			static_cast< TockWrapper* >( c->parent() )->
				processFuncLocal( info );
		}
		void processFuncLocal( ProcInfo info ) {
			std::ostringstream buf;
			buf << "Process: Tock " << label_ << "	t = "
			    << info->currTime_ << ", dt = " << info->dt_;
			info->setResponse( buf.str() );
		}
		static void reinitFunc( Conn* c ) {
			static_cast< TockWrapper* >( c->parent() )->
				reinitFuncLocal(  );
		}
		void reinitFuncLocal(  ) {
			; // cout << "Reinit: Tock " << label_ << "\n";
		}

///////////////////////////////////////////////////////
// Synapse creation and info access functions.       //
///////////////////////////////////////////////////////

///////////////////////////////////////////////////////
// Conn access functions.                            //
///////////////////////////////////////////////////////
		static Conn* getTickConn( Element* e ) {
			return &( static_cast< TockWrapper* >( e )->tickConn_ );
		}

///////////////////////////////////////////////////////
// Class creation and info access functions.         //
///////////////////////////////////////////////////////
		static Element* create(
			const string& name, Element* pa, const Element* proto ) {
			// Put tests for parent class here
			// Put proto initialization stuff here
			// const Tock* p = dynamic_cast<const Tock *>(proto);
			// if (p)... and so on. 
			return new TockWrapper(name);
		}

		const Cinfo* cinfo() const {
			return &cinfo_;
		}


    private:
///////////////////////////////////////////////////////
// MsgSrc template definitions.                      //
///////////////////////////////////////////////////////
		UniConn< tickConnLookup > tickConn_;

///////////////////////////////////////////////////////
// Synapse definition.                               //
///////////////////////////////////////////////////////

///////////////////////////////////////////////////////
// Static initializers for class and field info      //
///////////////////////////////////////////////////////
		static Finfo* fieldArray_[];
		static const Cinfo cinfo_;
};
#endif // _TockWrapper_h

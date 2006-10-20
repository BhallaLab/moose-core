using namespace std;

#ifndef _HSolveWrapper_h
#define _HSolveWrapper_h
class HSolveWrapper: 
	public HSolve, public Neutral
{
	friend Element* processConnHSolveLookup( const Conn* );
    public:
		HSolveWrapper(const string& n)
		:
			Neutral( n )
			// processConn uses a templated lookup function
		{
			;
		}
///////////////////////////////////////////////////////
//    EvalField header definitions.                  //
///////////////////////////////////////////////////////
		string localGetPath() const;
		static string getPath( const Element* e ) {
			return static_cast< const HSolveWrapper* >( e )->
			localGetPath();
		}
		void localSetPath( string value );
		static void setPath( Conn* c, string value ) {
			static_cast< HSolveWrapper* >( c->parent() )->
			localSetPath( value );
		}
///////////////////////////////////////////////////////
// Msgsrc header definitions .                       //
///////////////////////////////////////////////////////
///////////////////////////////////////////////////////
// dest header definitions .                         //
///////////////////////////////////////////////////////
		void reinitFuncLocal(  ) {
		}
		static void reinitFunc( Conn* c ) {
			static_cast< HSolveWrapper* >( c->parent() )->
				reinitFuncLocal(  );
		}

		void processFuncLocal( ProcInfo info );
		static void processFunc( Conn* c, ProcInfo info ) {
			static_cast< HSolveWrapper* >( c->parent() )->
				processFuncLocal( info );
		}


///////////////////////////////////////////////////////
// Synapse creation and info access functions.       //
///////////////////////////////////////////////////////

///////////////////////////////////////////////////////
// Conn access functions.                            //
///////////////////////////////////////////////////////
		static Conn* getProcessConn( Element* e ) {
			return &( static_cast< HSolveWrapper* >( e )->processConn_ );
		}

///////////////////////////////////////////////////////
// Class creation and info access functions.         //
///////////////////////////////////////////////////////
		static Element* create(
			const string& name, Element* pa, const Element* proto ) {
			// Put tests for parent class here
			// Put proto initialization stuff here
			// const HSolve* p = dynamic_cast<const HSolve *>(proto);
			// if (p)... and so on. 
			return new HSolveWrapper(name);
		}

		const Cinfo* cinfo() const {
			return &cinfo_;
		}


    private:
///////////////////////////////////////////////////////
// MsgSrc template definitions.                      //
///////////////////////////////////////////////////////
		UniConn< processConnHSolveLookup > processConn_;

///////////////////////////////////////////////////////
// Synapse definition.                               //
///////////////////////////////////////////////////////

///////////////////////////////////////////////////////
// Private functions and fields for the Wrapper class//
///////////////////////////////////////////////////////
	string path_;
	bool assignCompartmentParameters(ChCompartment& compt,Element* e );
	void configure();

///////////////////////////////////////////////////////
// Static initializers for class and field info      //
///////////////////////////////////////////////////////
		static Finfo* fieldArray_[];
		static const Cinfo cinfo_;
};
#endif // _HSolveWrapper_h

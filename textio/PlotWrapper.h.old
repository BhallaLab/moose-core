#ifndef _PlotWrapper_h
#define _PlotWrapper_h
class PlotWrapper: 
	public Plot, public Neutral
{
	friend Element* processConnPlotLookup( const Conn* );
	friend Element* trigPlotConnPlotLookup( const Conn* );
	friend Element* trigPlotInConnPlotLookup( const Conn* );
	friend Element* plotInConnPlotLookup( const Conn* );
	friend Element* printInConnPlotLookup( const Conn* );
    public:
		PlotWrapper(const string& n)
		:
			Neutral( n ),
			trigPlotSrc_( &trigPlotConn_ )
			// processConn uses a templated lookup function,
			// trigPlotConn uses a templated lookup function,
			// trigPlotInConn uses a templated lookup function,
			// plotInConn uses a templated lookup function,
			// printInConn uses a templated lookup function
		{
			;
		}
///////////////////////////////////////////////////////
//    Field header definitions.                      //
///////////////////////////////////////////////////////
		static void setCurrTime( Conn* c, double value ) {
			static_cast< PlotWrapper* >( c->parent() )->currTime_ = value;
		}
		static double getCurrTime( const Element* e ) {
			return static_cast< const PlotWrapper* >( e )->currTime_;
		}
		static void setPlotName( Conn* c, string value ) {
			static_cast< PlotWrapper* >( c->parent() )->plotName_ = value;
		}
		static string getPlotName( const Element* e ) {
			return static_cast< const PlotWrapper* >( e )->plotName_;
		}
		static void setNpts( Conn* c, int value ) {
			static_cast< PlotWrapper* >( c->parent() )->npts_ = value;
		}
		static int getNpts( const Element* e ) {
			return static_cast< const PlotWrapper* >( e )->npts_;
		}
		static void setJagged( Conn* c, int value ) {
			static_cast< PlotWrapper* >( c->parent() )->jagged_ = value;
		}
		static int getJagged( const Element* e ) {
			return static_cast< const PlotWrapper* >( e )->jagged_;
		}
		static void setX(
			Element* e, unsigned long index, double value );
		static double getX(
			const Element* e, unsigned long index );
		static void setY(
			Element* e, unsigned long index, double value );
		static double getY(
			const Element* e, unsigned long index );
///////////////////////////////////////////////////////
// Msgsrc header definitions .                       //
///////////////////////////////////////////////////////
		static SingleMsgSrc* getTrigPlotSrc( Element* e ) {
			return &( static_cast< PlotWrapper* >( e )->trigPlotSrc_ );
		}

///////////////////////////////////////////////////////
// dest header definitions .                         //
///////////////////////////////////////////////////////
		void reinitFuncLocal(  );
		static void reinitFunc( Conn* c ) {
			static_cast< PlotWrapper* >( c->parent() )->
				reinitFuncLocal(  );
		}

		void processFuncLocal( ProcInfo info );
		static void processFunc( Conn* c, ProcInfo info ) {
			static_cast< PlotWrapper* >( c->parent() )->
				processFuncLocal( info );
		}

		void trigPlotFuncLocal( double yval );
		static void trigPlotFunc( Conn* c, double yval ) {
			static_cast< PlotWrapper* >( c->parent() )->
				trigPlotFuncLocal( yval );
		}

		void plotFuncLocal( double yval );
		static void plotFunc( Conn* c, double yval ) {
			static_cast< PlotWrapper* >( c->parent() )->
				plotFuncLocal( yval );
		}

		void printFuncLocal( string fileName );
		static void printFunc( Conn* c, string fileName ) {
			static_cast< PlotWrapper* >( c->parent() )->
				printFuncLocal( fileName );
		}


///////////////////////////////////////////////////////
// Synapse creation and info access functions.       //
///////////////////////////////////////////////////////

///////////////////////////////////////////////////////
// Conn access functions.                            //
///////////////////////////////////////////////////////
		static Conn* getProcessConn( Element* e ) {
			return &( static_cast< PlotWrapper* >( e )->processConn_ );
		}
		static Conn* getTrigPlotConn( Element* e ) {
			return &( static_cast< PlotWrapper* >( e )->trigPlotConn_ );
		}
		static Conn* getTrigPlotInConn( Element* e ) {
			return &( static_cast< PlotWrapper* >( e )->trigPlotInConn_ );
		}
		static Conn* getPlotInConn( Element* e ) {
			return &( static_cast< PlotWrapper* >( e )->plotInConn_ );
		}
		static Conn* getPrintInConn( Element* e ) {
			return &( static_cast< PlotWrapper* >( e )->printInConn_ );
		}

///////////////////////////////////////////////////////
// Class creation and info access functions.         //
///////////////////////////////////////////////////////
		static Element* create(
			const string& name, Element* pa, const Element* proto ) {
			// Put tests for parent class here
			// Put proto initialization stuff here
			// const Plot* p = dynamic_cast<const Plot *>(proto);
			// if (p)... and so on. 
			return new PlotWrapper(name);
		}

		const Cinfo* cinfo() const {
			return &cinfo_;
		}


    private:
///////////////////////////////////////////////////////
// MsgSrc template definitions.                      //
///////////////////////////////////////////////////////
		SingleMsgSrc0 trigPlotSrc_;
		UniConn< processConnPlotLookup > processConn_;
		UniConn< trigPlotConnPlotLookup > trigPlotConn_;
		UniConn< trigPlotInConnPlotLookup > trigPlotInConn_;
		UniConn< plotInConnPlotLookup > plotInConn_;
		UniConn< printInConnPlotLookup > printInConn_;

///////////////////////////////////////////////////////
// Synapse definition.                               //
///////////////////////////////////////////////////////

///////////////////////////////////////////////////////
// Static initializers for class and field info      //
///////////////////////////////////////////////////////
		static Finfo* fieldArray_[];
		static const Cinfo cinfo_;
};
#endif // _PlotWrapper_h

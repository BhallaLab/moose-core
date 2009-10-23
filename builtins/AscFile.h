class AscFile
{
 public:
  AscFile();
  ~AscFile();

  ///////////////////////////////////////////////////
  // Dest functions
  ///////////////////////////////////////////////////
  static void processFunc( const Conn* c, ProcInfo p );
  void processFuncLocal(Eref e, double time);

  static void reinitFunc( const Conn* c, ProcInfo info );
  void reinitFuncLocal( Eref e );

  static void input( const Conn* c, double v );
  void inputLocal( unsigned int columnId, double v );

 /**
  * \todo At present this call must be made explicitly at the end of a simulation
  * to flush the file. Currently elements do not get destroyed at exit, which is
  * why the destructor also never gets called.
  */
  static void closeFunc( const Conn* c );
  void closeFuncLocal( );
  
  ///////////////////////////////////////////////////
  // Field access functions
  ///////////////////////////////////////////////////
  static string getFileName( Eref e );
  static void setFileName( const Conn* c, string s );

  static int getAppend( Eref e );
  static void setAppend( const Conn* c, int f );

  static int getTime( Eref e );
  static void setTime( const Conn* c, int time );

  static int getHeader( Eref e );
  static void setHeader( const Conn* c, int header );

  static string getDelimiter( Eref e );
  static void setDelimiter( const Conn* c, string delimiter );

  static string getComment( Eref e );
  static void setComment( const Conn* c, string comment );

 private:
  string filename_;
  int append_;
  int time_;
  int header_;
  string delimiter_;
  string comment_;
  
  vector<double> columnData_;
  /*
   * If fileOut_ is used as an object, instead of a pointer, then this file
   * does not compile. This is due to a call to the following function during
   * Cinfo initialization:
   *    ValueFtype1< AscFile >::global()
   * Not really sure why, but it has something to do with a copy constructor.
   */
  std::ofstream* fileOut_;
};

class AscFile
{
 public:
  AscFile();
  ~AscFile();

  static void processFunc( const Conn* c, ProcInfo p );
  void processFuncLocal(Eref e, double time);

  static void reinitFunc( const Conn* c, ProcInfo info );
  void reinitFuncLocal( const Conn* c );

  static void input( const Conn* c, double v );
  void inputLocal( unsigned int columnId, double v );

  static string getFileName( Eref e );
  static void setFileName( const Conn* c, string s );

  static int getAppendFlag( Eref e );
  static void setAppendFlag( const Conn* c, int f );

  void writeToFile();

 private:

  string fileName_;
  vector<double> columnData_;
  int nCols_;
  int appendFlag_;

  //std::ofstream *bup2; // If I pad here this one gets overwritten instead.
  std::ofstream *fileOut_;
  //std::ofstream *bup;
};

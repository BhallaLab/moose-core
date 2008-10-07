#ifndef _MOOSE_MUSIC_H
#define _MOOSE_MUSIC_H

class Music 
{

 public:
  Music() 
    {

    }

  //////////////////////////////////////////////////////////////////
  // Message dest functions.
  //////////////////////////////////////////////////////////////////
  
  // Call the MUSIC tick from here
  void innerProcessFunc( const Conn* c, ProcInfo p );
  static void processFunc( const Conn* c, ProcInfo p );
  
  static void reinitFunc( const Conn* c, ProcInfo p );

  static void addInputPort( const Conn* c, string name, 
                            string type, unsigned int width);

  static void addOutputPort( const Conn* c, string name, 
                             string type, unsigned int width);  

 protected:

 private:

  void innerAddInputPort( Eref e,  string name, 
                          string type, unsigned int width);
  void innerAddOutputPort( Eref e,  string name, 
                           string type, unsigned int width);

};








#endif // MOOSE_MUSIC_H

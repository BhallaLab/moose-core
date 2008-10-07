/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
**             and Niraj Dudani and Johannes Hjorth, KTH, Stockholm
**
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/

#include "moose.h"
#include "Music.h"
#include "element/Neutral.h"

const Cinfo* initMusicCinfo()
{

  /**
   * This is a shared message to receive Process messages from
   * the scheduler objects.
   */

  static Finfo* processShared[] =
    {
      new DestFinfo( "process", Ftype1< ProcInfo >::global(),
                     RFCAST( &Music::processFunc ) ),
      new DestFinfo( "reinit", Ftype1< ProcInfo >::global(),
                     RFCAST( &Music::reinitFunc ) ),
    };

  static Finfo* process = 
    new SharedFinfo( "process", processShared,
                     sizeof( processShared ) / sizeof( Finfo* ) );


  static Finfo* musicFinfos[] =
    {
      //////////////////////////////////////////////////////////////////
      // SharedFinfos
      //////////////////////////////////////////////////////////////////
      process,

      //////////////////////////////////////////////////////////////////
      // Dest Finfos.
      //////////////////////////////////////////////////////////////////
      new DestFinfo( "addInputPort", 
                     Ftype3< string, string, unsigned int >::global(),
                     RFCAST( &Music::addInputPort ) ),
      new DestFinfo( "addOutputPort", 
                     Ftype3< string, string, unsigned int >::global(),
                     RFCAST( &Music::addOutputPort ) ),



    };

  // CHECK WHEN THE TICK SHOULD BE CALLED
  static SchedInfo schedInfo[] = { { process, 0, 1 } };
  
  static Cinfo musicCinfo("Music",
                          "Niraj Dudani and Johannes Hjorth",
                          "Moose Music object for communciation with the MUSIC API",
                          initNeutralCinfo(),
                          musicFinfos,
                          sizeof( musicFinfos ) / sizeof( Finfo* ),
                          ValueFtype1< Music >::global(),
                          schedInfo, 1
                          );
  
  
  return &musicCinfo;

}

static const Cinfo* musicCinfo = initMusicCinfo();


void Music::innerProcessFunc( const Conn* c, ProcInfo p ) 
{
  cerr << "Music::innerProcessFunc not implemented yet" << endl;

}

void Music::processFunc( const Conn* c, ProcInfo p ) 
{
  static_cast < Music* > (c->data() )->innerProcessFunc(c,p);
}
  
void Music::reinitFunc( const Conn* c, ProcInfo p ) 
{

}


void Music::addInputPort( const Conn* c, string name, 
                          string type, unsigned int width) 
{

  static_cast < Music* > ( c->data() )->innerAddInputPort(c->target(), 
                                                          name, type, width);

}

void Music::addOutputPort( const Conn* c, string name, 
                           string type, unsigned int width) 
{
  static_cast < Music* > ( c->data() )->innerAddOutputPort(c->target(), 
                                                           name, type, width);

}

void Music::innerAddInputPort( Eref e,  string name, 
                               string type, unsigned int width) 
{
  Element* port = Neutral::create( "InputEventPort", name, 
                                   e.id(), Id::scratchId() );

  set<unsigned int>(port,"width", width);
}
 
void Music::innerAddOutputPort( Eref e,  string name, 
                                string type, unsigned int width) 
{

  cerr << "Music::innerAddOutputPort not implemented yet" << endl;
}








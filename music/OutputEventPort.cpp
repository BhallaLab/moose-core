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
#include <mpi.h>
#include <music.hh>
#include "Music.h"
#include "OutputEventPort.h"
#include "OutputEventChannel.h"
#include <sstream>
#include "element/Neutral.h"

const Cinfo* initOutputEventPortCinfo()
{

  static Finfo* processShared[] =
    {
      new DestFinfo( "process", Ftype1< ProcInfo >::global(),
                     dummyFunc ),
      new DestFinfo( "reinit", Ftype1< ProcInfo >::global(),
                     RFCAST( &OutputEventPort::reinitFunc ) )
    };

  static Finfo* process = 
    new SharedFinfo( "process", processShared,
                     sizeof( processShared ) / sizeof( Finfo* ) );




  static Finfo* outputEventPortFinfos[] =
    {
      new ValueFinfo( "width", ValueFtype1< unsigned int >::global(),
                      GFCAST( &OutputEventPort::getWidth ),
                      &dummyFunc
                      ),
      new ValueFinfo( "maxBuffered", ValueFtype1< int >::global(),
                      GFCAST( &OutputEventPort::getMaxBuffered ),
                      RFCAST( &OutputEventPort::setMaxBuffered )
                      ),
      new DestFinfo( "initialise", 
                     Ftype3< unsigned int, unsigned int,
                             MUSIC::event_output_port* >::global(),
                     RFCAST( &OutputEventPort::initialiseFunc )
                     ),
      process
    };

  
  static Cinfo outputEventPortCinfo("OutputEventPort",
                                    "Niraj Dudani and Johannes Hjorth",
                                    "OutputEventPort for communciation with the MUSIC API",
                                    initNeutralCinfo(),
                                    outputEventPortFinfos,
                                    sizeof( outputEventPortFinfos ) / sizeof( Finfo* ),
                                    ValueFtype1< OutputEventPort >::global() );
  
  
  return &outputEventPortCinfo;

}

static const Cinfo* outputEventPortCinfo = initOutputEventPortCinfo();



void OutputEventPort::reinitFunc( const Conn* c, ProcInfo p ) 
{
  static_cast < OutputEventPort* > (c->data())->innerReinitFunc();

}

void OutputEventPort::innerReinitFunc() 
{
  // Map the output from MUSIC to data channels local to this process
cerr << "Port connected? " << mPort_->is_connected() << endl;
  MUSIC::linear_index iMap(myOffset_, myWidth_);
  mPort_->map(&iMap, maxBuffered_);

}



void OutputEventPort::initialiseFunc( const Conn* c, 
                                      unsigned int width,
                                      unsigned int offset,
                                      MUSIC::event_output_port* mPort)
{
  static_cast < OutputEventPort* > 
    (c->data())->innerInitialiseFunc(c->target(), width, offset, mPort);
}

void OutputEventPort::innerInitialiseFunc( Eref e, 
                                           unsigned int width, 
                                           unsigned int offset,
                                           MUSIC::event_output_port* 
                                                 mPort) 
{

  myWidth_ = width;
  myOffset_ = offset;

  mPort_ = mPort;

  for(unsigned int i = channels_.size(); i < width; i++)
    {
      ostringstream name;

      name << "channel[" << i + offset << "]";

      Element* channel = Neutral::create( "OutputEventChannel", name.str(),
                                          e.id(), Id::scratchId() );
      channels_.push_back(channel->id());

      set< unsigned int, MUSIC::event_output_port* > 
        (channel, "initialise", i, mPort);
    }

}

unsigned int OutputEventPort::getWidth( Eref e ) 
{
  return static_cast < OutputEventPort* > (e.data())->channels_.size();
}

int OutputEventPort::getMaxBuffered( Eref e )
{
  return static_cast < OutputEventPort* > (e.data())->maxBuffered_;
}

void OutputEventPort::setMaxBuffered( const Conn* c, int maxBuffered )
{
  static_cast < OutputEventPort* > (c->data())->maxBuffered_ = maxBuffered;
}


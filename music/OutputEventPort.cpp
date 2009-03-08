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
      new ValueFinfo( "isConnected", ValueFtype1< unsigned int >::global(),
                      GFCAST( &OutputEventPort::getIsConnected ),
                      &dummyFunc
                      ),
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
                             MUSIC::EventOutputPort* >::global(),
                     RFCAST( &OutputEventPort::initialiseFunc )
                     ),
      process
    };
  
  /**
   * Music ports should be initialized before the MusicManager gets initialized.
   * In the default autoschedule,
   * 
   * By default, /music is connected to clock 0, stage 1. In some cases it is
   * possible to attach /music to a slow clock.
   */
  static SchedInfo schedInfo[] = { { process, 0, 0 } };
  
  static string doc[] =
	{
		"Name", "OutputEventPort",
		"Author", "Niraj Dudani and Johannes Hjorth",
		"Description", "OutputEventPort for communciation with the MUSIC API",
	};

  static Cinfo outputEventPortCinfo(
                                    doc,
		 		    sizeof( doc ) / sizeof( string ),
                                    initNeutralCinfo(),
                                    outputEventPortFinfos,
                                    sizeof( outputEventPortFinfos ) / sizeof( Finfo* ),
                                    ValueFtype1< OutputEventPort >::global(),
                                    schedInfo, 1 );

  
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
//~ cerr << "Port connected? " << mPort_->isConnected() << endl;

// Add check if the config file specifies it as isConnected()?
  if(!isMapped_) { 
    MUSIC::LinearIndex iMap(myOffset_, myWidth_);
    mPort_->map(&iMap, MUSIC::Index::LOCAL, maxBuffered_);
    isMapped_ = 1;
  }

}



void OutputEventPort::initialiseFunc( const Conn* c, 
                                      unsigned int width,
                                      unsigned int offset,
                                      MUSIC::EventOutputPort* mPort)
{
  static_cast < OutputEventPort* > 
    (c->data())->innerInitialiseFunc(c->target(), width, offset, mPort);
}

void OutputEventPort::innerInitialiseFunc( Eref e, 
                                           unsigned int width, 
                                           unsigned int offset,
                                           MUSIC::EventOutputPort* 
                                                 mPort) 
{

  myWidth_ = width;
  myOffset_ = offset;

  mPort_ = mPort;

  for(unsigned int i = 0; i < width; i++)
    {
      ostringstream name;

      name << "channel[" << i + offset << "]";

      Element* channel = Neutral::create( "OutputEventChannel", name.str(),
                                          e.id(), Id::scratchId() );
      channels_.push_back(channel);

      set< unsigned int, MUSIC::EventOutputPort* > 
        (channel, "initialise", i, mPort);
    }

}

unsigned int OutputEventPort::getIsConnected( Eref e ) 
{
	return static_cast < OutputEventPort* > (e.data())->
		mPort_->isConnected();
}

unsigned int OutputEventPort::getWidth( Eref e ) 
{
	return static_cast < OutputEventPort* > (e.data())->
		mPort_->width();
}

int OutputEventPort::getMaxBuffered( Eref e )
{
  return static_cast < OutputEventPort* > (e.data())->maxBuffered_;
}

void OutputEventPort::setMaxBuffered( const Conn* c, int maxBuffered )
{
  static_cast < OutputEventPort* > (c->data())->maxBuffered_ = maxBuffered;
}


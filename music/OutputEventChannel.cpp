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
#include <music.hh>
// #include "Music.h"
#include "OutputEventChannel.h"

const Cinfo* initOutputEventChannelCinfo()
{


  static Finfo* outputEventChannelFinfos[] =
    {
      //////////////////////////////////////////////////////////////////
      // SharedFinfos
      //////////////////////////////////////////////////////////////////
      // process,

      ///////////////////////////////////////////////////////
      // MsgSrc definitions
      ///////////////////////////////////////////////////////
	  new DestFinfo( "synapse", Ftype1< double >::global() ,
                     RFCAST( &OutputEventChannel::insertEvent),
					 "This field receives event messages in the form of time of an action "
					 "potential.It is called 'synapse' because a similar field on SynChan "
					 "objects is also called synapse." ),      
      new DestFinfo("initialise", 
                    Ftype2< unsigned int, MUSIC::EventOutputPort* >::global(),
                    RFCAST( &OutputEventChannel::initialise))

    };

  static string doc[] =
	{
		"Name", "OutputEventChannel",
		"Author", "Niraj Dudani and Johannes Hjorth",
		"Description", "OutputEventChannel for communciation with the MUSIC API",
	};

  static Cinfo outputEventChannelCinfo(
                                      doc,
		                      sizeof( doc ) / sizeof( string ),                                      
				      initNeutralCinfo(),
                                      outputEventChannelFinfos,
                                      sizeof( outputEventChannelFinfos ) / sizeof( Finfo* ),
                                      ValueFtype1< OutputEventChannel >::global() );
  
  
  return &outputEventChannelCinfo;

}

static const Cinfo* outputEventChannelCinfo = initOutputEventChannelCinfo();

void OutputEventChannel::insertEvent(const Conn* c, double time) 
{
  static_cast < OutputEventChannel* > (c->data())->innerInsertEvent(time);
}

void OutputEventChannel::innerInsertEvent(double time)
{
  mPort_->insertEvent(time, localId_);
  //~ cerr << "event sent @ " << time << endl;
  //~ cerr << "id: " << localId_ << endl;
}

void OutputEventChannel::initialise(const Conn* c, unsigned int id, 
                                    MUSIC::EventOutputPort* mPort) 
{
  static_cast < OutputEventChannel* > (c->data())->localId_ 
    = MUSIC::LocalIndex(id);
  static_cast < OutputEventChannel* > (c->data())->mPort_ = mPort;
}

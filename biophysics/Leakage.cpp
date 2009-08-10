// Leakage.cpp --- 
// 
// Filename: Leakage.cpp
// Description: 
// Author: subhasis ray
// Maintainer: 
// Created: Mon Aug  3 02:32:29 2009 (+0530)
// Version: 
// Last-Updated: Mon Aug 10 11:15:39 2009 (+0530)
//           By: subhasis ray
//     Update #: 71
// URL: 
// Keywords: 
// Compatibility: 
// 
// 

// Commentary: Reimplementation of leakage class of GENESIS
// 
// 
// 
// 

// Change log:
// 
// 
// 
// 
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 3, or
// (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with this program; see the file COPYING.  If not, write to
// the Free Software Foundation, Inc., 51 Franklin Street, Fifth
// Floor, Boston, MA 02110-1301, USA.
// 
// 

// Code:

#include "moose.h"
#include "element/Neutral.h"
#include "shell/Shell.h"
#include "Leakage.h"

const Cinfo* initLeakageCinfo()
{
    static Finfo* processShared[] = {
        new DestFinfo("process", Ftype1<ProcInfo>::global(),
                      RFCAST( &Leakage::processFunc )),
        new DestFinfo("reinit", Ftype1<ProcInfo>::global(),
                      RFCAST( &Leakage::reinitFunc )),
    };
    static Finfo* process = new SharedFinfo("process", processShared,
                                            sizeof(processShared) / sizeof(Finfo*),
                                            "This is a shared message to receive Process message from the scheduler. "
                                            "The first entry is a MsgDest for the Process operation. It has a single argument, ProcInfo, which "
                                            "holds lots of information about current time, thread, dt and so on.\n"
                                            "The second entry is a MsgDest for the Reinit operation. It also uses ProcInfo." );
    static Finfo* channelShared[] = {
        new SrcFinfo("channel", Ftype2<double, double>::global()),
        new DestFinfo("Vm", Ftype1<double>::global(),
                      RFCAST(&Leakage::channelFunc)),
    };
    static Finfo* LeakageFinfos[] = {
        new ValueFinfo( "Ek", ValueFtype1< double >::global(),
			GFCAST( &Leakage::getEk ), 
			RFCAST( &Leakage::setEk ),
                        "leakage battery voltage."),
        new ValueFinfo( "Gk", ValueFtype1< double >::global(),
                        GFCAST( &Leakage::getGk ), 
                        RFCAST( &Leakage::setGk ),
                        "Leakage conductance."),
        new ValueFinfo( "Ik", ValueFtype1< double >::global(),
			GFCAST( &Leakage::getIk ), 
                        &dummyFunc,
                        "Leakage current."),
        new ValueFinfo( "activation", ValueFtype1< double >::global(),
			GFCAST( &Leakage::getActivation ), 
			RFCAST( &Leakage::setActivation ),
                        "Leakage conductance used for calculating Ik. Can be different from Gk"),
        process,
        new SharedFinfo("channel", channelShared,
                        sizeof(channelShared)/sizeof(Finfo*)),
        new SrcFinfo( "IkSrc", Ftype1< double >::global() ),

    };
    
    static SchedInfo schedInfo[] = { { process, 0, 1 } };
    
    static string doc[] = {
        "Name", "Leakage",
        "Author", "Subhasis Ray, 2009, NCBS",
        "Description", "Leakage: Passive leakage channel."
    };
    
    static Cinfo LeakageCinfo(
            doc,
            sizeof( doc ) / sizeof( string ),
            initNeutralCinfo(),
            LeakageFinfos,
            sizeof( LeakageFinfos )/sizeof(Finfo *),
            ValueFtype1< Leakage >::global(),
            schedInfo, 1);

    return &LeakageCinfo;
}

static const Cinfo* leakageCinfo = initLeakageCinfo();

//////// Slots ///////////
static const Slot channelSlot =
	initLeakageCinfo()->getSlot( "channel.channel" );
static const Slot ikSlot =
	initLeakageCinfo()->getSlot( "IkSrc" );

//////// Function definitions
void Leakage::setEk( const Conn* c, double Ek )
{
	static_cast< Leakage* >( c->data() )->Ek_ = Ek;
}
double Leakage::getEk( Eref e )
{
	return static_cast< Leakage* >( e.data() )->Ek_;
}

void Leakage::setGk( const Conn* c, double Gk )
{
	static_cast< Leakage* >( c->data() )->Gk_ = Gk;
}
double Leakage::getGk( Eref e )
{
	return static_cast< Leakage* >( e.data() )->Gk_;
}
void Leakage::setActivation( const Conn* c, double activation )
{
	static_cast< Leakage* >( c->data() )->activation_ = activation;
}
double Leakage::getActivation( Eref e )
{
	return static_cast< Leakage* >( e.data() )->activation_;
}

double Leakage::getIk( Eref e )
{
	return static_cast< Leakage* >( e.data() )->Ik_;
}
void Leakage::processFunc( const Conn* c, ProcInfo p )
{
    Eref e = c->target();
    Leakage* object = static_cast< Leakage* >( c->data() );
    send2< double, double >( e, channelSlot, object->Gk_, object->Ek_ );
    object->Ik_ = (object->Ek_ - object->Vm_) * object->Gk_;
    send1< double >( e, ikSlot, object->Ik_ );
}
void Leakage::reinitFunc( const Conn* c, ProcInfo p )
{
    Leakage* object = static_cast< Leakage* >( c->data() );
    send2< double, double >( c->target(), channelSlot, object->Gk_, object->Ek_ );
    object->Ik_ = 0.0;
}
void Leakage::channelFunc( const Conn* c, double Vm )
{
	static_cast< Leakage* >( c->data() )->Vm_ = Vm;
}

// 
// Leakage.cpp ends here

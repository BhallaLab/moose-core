#ifndef _Zombie_HHChannel_h
#define _Zombie_HHChannel_h
/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2003-20012 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
*********************************************************************
*/

/**
 * Zombie object that lets HSolve do its calculations, while letting the user
 * interact with this object as if it were the original object.
 *
 * ZombieHHChannel derives directly from Neutral, unlike the regular
 * HHChannel which derives from ChanBase. ChanBase handles fields like
 * Gbar, Gk, Ek, Ik, which are common to HHChannel, SynChan, etc. On the
 * other hand, these fields are stored separately for HHChannel and SynChan
 * in the HSolver. Hence we cannot have a ZombieChanBase which does, for
 * example:
 *           hsolve_->setGk( id, Gk );
 * Instead we must have ZombieHHChannel and ZombieSynChan which do:
 *           hsolve_->setHHChannelGk( id, Gk );
 * and:
 *           hsolve_->setSynChanGk( id, Gk );
 * respectively.
 */

#include "../basecode/header.h"
#include "../basecode/ElementValueFinfo.h"
#include "HinesMatrix.h"
#include "HSolveStruct.h"
#include "HSolvePassive.h"
#include "RateLookup.h"
#include "HSolveActive.h"
#include "HSolve.h"
#include "../biophysics/HHGate.h"
#include "../biophysics/ChanBase.h"
#include "../biophysics/ChanCommon.h"
#include "../biophysics/Compartment.h"
#include "../biophysics/HHChannelBase.h"
#include "../biophysics/HHChannel.h"
#include "../utility/print_function.hpp"

class ZombieHHChannel: public HHChannelBase
{
public:
    ZombieHHChannel();

    /////////////////////////////////////////////////////////////
    // Value field access function definitions
    /////////////////////////////////////////////////////////////

    void vSetGbar( const Eref& e , double Gbar ) override;
    double vGetGbar( const Eref& e  ) const override;
    void vSetGk( const Eref& e , double Gk ) override;
    double vGetGk( const Eref& e  ) const override;
    void vSetEk( const Eref& e , double Ek ) override;
    double vGetEk( const Eref& e  ) const override;
    void vSetIk( const Eref& e, double Ik ) override;
    double vGetIk( const Eref& e  ) const override;
    void vSetXpower( const Eref& e , double Xpower ) override;
    void vSetYpower( const Eref& e , double Ypower ) override;
    void vSetZpower( const Eref& e , double Zpower ) override;
    void vSetInstant( const Eref& e , int instant ) override;
    int vGetInstant( const Eref& e  ) const override;
    void vSetX( const Eref& e , double X ) override;
    double vGetX( const Eref& e  ) const override;
    void vSetY( const Eref& e , double Y ) override;
    double vGetY( const Eref& e  ) const override;
    void vSetZ( const Eref& e , double Z ) override;
    double vGetZ( const Eref& e  ) const override;
    /**
     * Not trivial to change Ca-dependence once HSolve has been set up, and
     * unlikely that one would want to change this field after setup, so
     * keeping this field read-only.
     */
    void vSetUseConcentration( const Eref& e, int value ) override;
    // implemented in baseclass: int getUseConcentration() const;

    void vSetModulation( const Eref& e, double value ) override;

    /////////////////////////////////////////////////////////////
    // Dest function definitions
    /////////////////////////////////////////////////////////////

    void vProcess( const Eref& e, ProcPtr p ) override;
    void vReinit( const Eref& e, ProcPtr p ) override;
    void vHandleConc( const Eref& e, double value) override;
    void vCreateGate(const Eref& e , string name) override;

    /////////////////////////////////////////////////////////////
	// Dummy function, not needed in Zombie.
	void vHandleVm( double Vm ) override;

    /////////////////////////////////////////////////////////////
    // Gate handling functions
    /////////////////////////////////////////////////////////////
    // /**
    //  * Access function used for the X gate. The index is ignored.
    //  */
    // HHGate* vGetXgate( unsigned int i ) const override;

    // /**
    //  * Access function used for the Y gate. The index is ignored.
    //  */
    // HHGate* vGetYgate( unsigned int i ) const override;

    // /**
    //  * Access function used for the Z gate. The index is ignored.
    //  */
    // HHGate* vGetZgate( unsigned int i ) const override;
    /////////////////////////////////////////////////////////////
	void vSetSolver( const Eref& e , Id hsolve ) override;

    static const Cinfo* initCinfo();

private:
    HSolve* hsolve_;

    void copyFields( Id chanId, HSolve* hsolve_ );
};


#endif // _Zombie_HHChannel_h

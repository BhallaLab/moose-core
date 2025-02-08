/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           copyright (C) 2003-2011 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#ifndef _HHGateBase_h
#define _HHGateBase_h

/**
 * This class is the base for gates in Hodgkin-Huxley type channels
 * (HHGates). HHGates are typically shared. This means that when you
 * make a copy or a vector of an HHChannel, there is only a single
 * HHGate created, and its pointer is used by all the copies.  The
 * lookup functions are thread-safe.  Field assignment to the HHGate
 * should be possible only from the original HHChannel, but all the
 * others do have read permission.
 * This base class only factors out the tracking of the ownership.
 */

class HHGateBase {
public:
    /**
     * Dummy constructor, to keep Dinfo happy. Should never be used
     */
    HHGateBase();

    /**
     * This constructor is the one meant to be used. It takes the
     * originalId of the parent HHChannel as a required argument,
     * so that any subsequent 'write' functions can be checked to
     * see if they are legal. Also tracks its own Id.
     */
    HHGateBase(Id originalChanId, Id originalGateId);

    /////////////////////////////////////////////////////////////////
    // Utility funcs
    /////////////////////////////////////////////////////////////////
    /**
     * Checks if the provided Id is the one that the HHGate was created
     * on. If true, fine, otherwise complains about trying to set the
     * field.
     */
    bool checkOriginal(Id id, const string& field) const;

    /**
     * isOriginalChannel returns true if the provided Id is the Id of
     * the channel on which the HHGate was created.
     */
    bool isOriginalChannel(Id id) const;

    /**
     * isOriginalChannel returns true if the provided Id is the Id of
     * the Gate created at the same time as the original channel.
     */
    bool isOriginalGate(Id id) const;

    /**
     * Returns the Id of the original Channel.
     */
    Id originalChannelId() const;

    /**
     * Returns the Id of the original Gate.
     */
    Id originalGateId() const;

    static const Cinfo* initCinfo();

protected:
    /**
     * Id of original channel, the one which has actually allocated it,
     * All other Elements have to treat the values as readonly.
     */
    Id originalChanId_;

    /**
     * Id of original Gate, the one which was created with the original
     * channel.
     * All other Elements have to treat the values as readonly.
     */
    Id originalGateId_;
};

#endif  // _HHGateBase_h

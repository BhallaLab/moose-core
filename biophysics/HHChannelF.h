/* HHChannelF.h --- 
 * 
 * Filename: HHChannelF.h
 * Description: 
 * Author: Subhasis Ray
 * Created: Fri Jan 24 13:09:34 2025 (+0530)
 */

#ifndef _HHChannelF_h
#define _HHChannelF_h

#include "HHChannelBase.h"

class HHGateF;

class HHChannelF : public HHChannelBase {
#ifdef DO_UNIT_TESTS
    /* friend void testHHChannelF(); */
    /* friend void testHHGateFCreation(); */
#endif
public:
    HHChannelF();
    ~HHChannelF();
    void innerSetXpower(double Xpower);
    void innerSetYpower(double Ypower);
    void innerSetZpower(double Zpower);
    /////////////////////////////////////////////////////////////
    // Dest function definitions
    /////////////////////////////////////////////////////////////

    /**
     * processFunc handles the update and calculations every
     * clock tick. It first sends the request for evaluation of
     * the gate variables to the respective gate objects and
     * recieves their response immediately through a return
     * message. This is done so that many channel instances can
     * share the same gate lookup tables, but do so cleanly.
     * Such messages should never go to a remote node.
     * Then the function does its own little calculations to
     * send back to the parent compartment through regular
     * messages.
     */
    void vProcess(const Eref& e, ProcPtr p) override;

    /**
     * Reinitializes the values for the channel. This involves
     * computing the steady-state value for the channel gates
     * using the provided Vm from the parent compartment. It
     * involves a similar cycle through the gates and then
     * updates to the parent compartment as for the processFunc.
     */
    void vReinit(const Eref& e, ProcPtr p) override;

    /**
     * Assign the local Vm_ to the incoming Vm from the compartment
    void handleVm( double Vm );
     */

    /**
     * Assign the local conc_ to the incoming conc from the
     * concentration calculations for the compartment. Typically
     * the message source will be a CaConc object, but there
     * are other options for computing the conc.
     */
    void vHandleConc(const Eref& e, double conc) override;

    /////////////////////////////////////////////////////////////
    // Gate handling functions
    /////////////////////////////////////////////////////////////
    HHGateF* getXgate(unsigned int i);
    HHGateF* getYgate(unsigned int i);
    HHGateF* getZgate(unsigned int i);

    void setNumGates(unsigned int num);
    unsigned int getNumXgates() const;
    unsigned int getNumYgates() const;
    unsigned int getNumZgates() const;

    /// Inner utility function for creating the gate.
    void innerCreateGate(const string& gateName, HHGateF** gatePtr, Id chanId,
                         Id gateId);

    /// Returns true if channel is original, false if copy.
    bool checkOriginal(Id chanId) const override;

    void vCreateGate(const Eref& e, string gateType) override;
    void destroyGate(const Eref& e, string gateType) override;

    /**
     * Inner utility for destroying the gate
     */
    void innerDestroyGate(const string& gateName, HHGateF** gatePtr, Id chanId);

    // /**
    //  * Utility for altering gate powers
    //  */
    // bool setGatePower(const Eref& e, double power, double* assignee,
    //                   const string& gateType);

    /////////////////////////////////////////////////////////////
    static const Cinfo* initCinfo();

   private:
    /// Conc_ is input variable for Ca-dependent channels.
    double conc_;

    /**
     * HHGate data structure for the xGate. This is writable only
     * on the HHChannel that originally created the HHGate, for others
     * it must be treated as readonly.
     */
    HHGateF* xGate_;

    /// HHGate data structure for the yGate.
    HHGateF* yGate_;

    /// HHGate data structure for the yGate.
    HHGateF* zGate_;

};

#endif

/* HHChannelF.h ends here */

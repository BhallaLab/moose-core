/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment,
** also known as GENESIS 3 base code.
**           copyright (C) 2014 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
*********************************************************************
*/

#ifndef _HHChannelBase_h
#define _HHChannelBase_h

#include "ChanCommon.h"

typedef double (*PFDD)(double, double);

class HHGate;

/**
 * The HHChannelBase is the base class for defining Hodgkin-Huxley type
 * channels, specifically dealing with derivatives used in the HSolver.
 * This is derived from the ChanBase rather than the ChanCommon, since
 * the Zombie classes used in the HSolver will not use the ChanCommon
 * fields.
 */

class HHChannelBase : public ChanCommon {
public:
    HHChannelBase();
    virtual ~HHChannelBase() = 0;  // this class is not to be instantiated

    /////////////////////////////////////////////////////////////
    // Value field access function definitions
    /////////////////////////////////////////////////////////////

    void setXpower(const Eref& e, double Xpower);
    double getXpower(const Eref& e) const;
    void setYpower(const Eref& e, double Ypower);
    double getYpower(const Eref& e) const;
    void setZpower(const Eref& e, double Zpower);
    double getZpower(const Eref& e) const;
    void setInstant(const Eref& e, int Instant);
    int getInstant(const Eref& e) const;
    void setX(const Eref& e, double X);
    double getX(const Eref& e) const;
    void setY(const Eref& e, double Y);
    double getY(const Eref& e) const;
    void setZ(const Eref& e, double Z);
    double getZ(const Eref& e) const;
    void setUseConcentration(const Eref& e, int value);
    int getUseConcentration(const Eref& e) const;
    // double vGetModulation( const Eref& e ) const; // provided by ChanCommon
    /////////////////////////////////////////////////////////////
    // Dest function definitions
    /////////////////////////////////////////////////////////////
    /**
     * Assign the local conc_ to the incoming conc from the
     * concentration calculations for the compartment. Typically
     * the message source will be a CaConc object, but there
     * are other options for computing the conc.
     */
    void handleConc(const Eref& e, double conc);

    /////////////////////////////////////////////////////////////
    // Gate handling functions
    /////////////////////////////////////////////////////////////
    /**
     * Access function used for the X gate. The index is ignored.
     */
    HHGate* getXgate(unsigned int i);

    /**
     * Access function used for the Y gate. The index is ignored.
     */
    HHGate* getYgate(unsigned int i);

    /**
     * Access function used for the Z gate. The index is ignored.
     */
    HHGate* getZgate(unsigned int i);

    /**
     * Dummy assignment function for the number of gates.
     */
    void setNumGates(unsigned int num);

    /**
     * Access function for the number of Xgates. Gives 1 if present,
     * otherwise 0.
     */
    // unsigned int getNumXgates() const;
    // /// Returns 1 if Y gate present, otherwise 0
    // unsigned int getNumYgates() const;
    // /// Returns 1 if Z gate present, otherwise 0
    // unsigned int getNumZgates() const;

    /**
     * Function for safely creating each gate, identified by strings
     * as X, Y and Z. Will only work on a new channel, not on a
     * copy. The idea is that the gates are always referred to the
     * original 'library' channel, and their contents cannot be touched
     * except by the original.
     */
    void createGate(const Eref& e, string gateType);
    /**
     * Utility function for destroying gate. Works only on original
     * HHChannel. Somewhat dangerous, should never be used after a
     * copy has been made as the pointer of the gate will be in use
     * elsewhere.
     * This needs to be virtual as it needs to call subclass specific delete
     * methods for the gate pointer.
     */
    virtual void destroyGate(const Eref& e, string gateType);
    /**
     * Utility for altering gate powers
     */
    bool setGatePower(const Eref& e, double power, double* assignee,
                      const string& gateType);

    /////////////////////////////////////////////////////////////
    // Virtual Value field access function definitions
    /////////////////////////////////////////////////////////////
    virtual void vSetXpower(const Eref& e, double Xpower);
    virtual void vSetYpower(const Eref& e, double Ypower);
    virtual void vSetZpower(const Eref& e, double Zpower);
    // getXpower etc functions are implemented here in the baseclass.
    virtual void vSetInstant(const Eref& e, int Instant);
    virtual int vGetInstant(const Eref& e) const;
    virtual void vSetX(const Eref& e, double X);
    virtual double vGetX(const Eref& e) const;
    virtual void vSetY(const Eref& e, double Y);
    virtual double vGetY(const Eref& e) const;
    virtual void vSetZ(const Eref& e, double Z);
    virtual double vGetZ(const Eref& e) const;
    virtual void vSetUseConcentration(const Eref& e, int value);
    virtual bool checkOriginal(Id chanId) const;
    /////////////////////////////////////////////////////////////
    // Some more Virtual Value field functions from ChanBase,
    // to be defined in derived classes. Listed here for clarity.
    /////////////////////////////////////////////////////////////
    // void vSetGbar( double Gbar );
    // double vGetGbar() const;
    // void vSetEk( double Ek );
    // double vGetEk() const;
    // void vSetGk( double Gk );
    // double vGetGk() const;
    // void vSetIk( double Ic );
    // double vGetIk() const;
    // void vHandleVm( double Vm );

    /////////////////////////////////////////////////////////////
    // Virtual Dest function definitions
    /////////////////////////////////////////////////////////////
    // void vProcess( const Eref& e, ProcPtr p ); // Listed for clarity
    // void vReinit( const Eref& e, ProcPtr p ); // Listed for clarity

    virtual void vHandleConc(const Eref& e, double conc);

    /////////////////////////////////////////////////////////////
    // Virtual Gate handling functions
    /////////////////////////////////////////////////////////////
    // virtual HHGate* vGetXgate(unsigned int i) const;
    // virtual HHGate* vGetYgate(unsigned int i) const;
    // virtual HHGate* vGetZgate(unsigned int i) const;
    virtual void vCreateGate(const Eref& e, string gateType) = 0;
    /////////////////////////////////////////////////////////////
    // Utility functions for taking integer powers.
    /////////////////////////////////////////////////////////////
    static double power1(double x, double p)
    {
        return x;
    }
    static double power2(double x, double p)
    {
        return x * x;
    }
    static double power3(double x, double p)
    {
        return x * x * x;
    }
    static double power4(double x, double p)
    {
        return power2(x * x, p);
    }
    static double powerN(double x, double p);

    static PFDD selectPower(double power);

    /////////////////////////////////////////////////////////////
    // Zombification functions.
    /////////////////////////////////////////////////////////////
    virtual void vSetSolver(const Eref& e, Id hsolve);
    static void zombify(Element* orig, const Cinfo* zClass, Id hsolve);

    /////////////////////////////////////////////////////////////
    static const Cinfo* initCinfo();

protected:
    double integrate(double state, double dt, double A, double B);
    /// Function pointers to specific power calculation function. This
    /// is an optimization to avoid the full pow function for simple
    /// commonly occurring integer exponents
    double (*takeXpower_)(double, double);
    double (*takeYpower_)(double, double);
    double (*takeZpower_)(double, double);
    /// Exponent for X gate
    double Xpower_;
    /// Exponent for Y gate
    double Ypower_;
    /// Exponent for Z gate
    double Zpower_;
    /// Flag for use of conc for input to Z gate calculations.
    bool useConcentration_;
    /// Flag to indicate if gates have been created
    bool xInited_, yInited_, zInited_;
    /// Bitmasked flag indicating which gates are to be computed as
    /// instantaneous (i.e., no integration)
    int instant_;

    double X_, Y_, Z_;
    double g_;  /// conductance without modulation component is different from
                /// Gk_ in ChanCommon

    Id myId_;

    static const double EPSILON;
    static const int INSTANT_X;
    static const int INSTANT_Y;
    static const int INSTANT_Z;
    /// Value used to scale channel conductance up or down
    // double modulation_; // this clashes with same field in ChanCommon
};

#endif  // _HHChannelBase_h

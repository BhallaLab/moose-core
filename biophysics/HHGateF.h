/**********************************************************************
** This program is part of 'MOOSE', the
** Messaging Object Oriented Simulation Environment.
**           copyright (C) 2003-2011 Upinder S. Bhalla. and NCBS
** It is made available under the terms of the
** GNU Lesser General Public License version 2.1
** See the file COPYING.LIB for the full notice.
**********************************************************************/
#ifndef _HHGateF_h
#define _HHGateF_h

#include "exprtk.hpp"
#include "../basecode/global.h"

#include "../basecode/header.h"
#include "../basecode/ElementValueFinfo.h"
#include "HHGateBase.h"

/**
 * This class handles a single gate on an HHChannel. It is equivalent to the
 * m and h terms on the Hodgkin-Huxley Na channel, or the n term on the
 * K channel. It stores the
 * voltage-dependence (sometimes concentration-dependence) of the
 * gating variables for opening the channel. It does so in a tabular form
 * which can be directly filled using experimental data points.
 * It also provides a set of
 * utility functions for defining the gate in functional forms, and
 * accessing those original functional forms.
 * The HHGateF is
 * accessed as a FieldElement, which means that it is available as a
 * pointer on the HHChannel. HHGateFs are typically shared. This means that
 * when you make a copy or a vector of an HHChannel, there is only a single
 * HHGateF created, and its pointer is used by all the copies.
 * The lookup functions are thread-safe.
 * Field assignment to the HHGateF should be possible only from the
 * original HHChannel, but all the others do have read permission.
 * Whereas HHGate uses interpolation tables, HHGateF uses direct
 * formula evaluation, hence slower but possibly more accurate.
 */

class HHGateF : public HHGateBase {
public:
    /**
     * Dummy constructor, to keep Dinfo happy. Should never be used
     */
    HHGateF();

    /**
     * This constructor is the one meant to be used. It takes the
     * originalId of the parent HHChannel as a required argument,
     * so that any subsequent 'write' functions can be checked to
     * see if they are legal. Also tracks its own Id.
     */
    HHGateF(Id originalChanId, Id originalGateId);
    /// HHGates remain shared between copies of a channel, so it
    /// should never be copied. Yet we need to define this because
    /// eprtk parser deletes its copy assignment, which deletes
    /// compiler generated copy assignment operator for HHGateF, which
    /// raises error with Dinfo, which tries to reference the copy
    /// operator.
    HHGateF& operator=(const HHGateF&); 
    //////////////////////////////////////////////////////////
    // LookupValueFinfos
    //////////////////////////////////////////////////////////
    /**
     * lookupA: Look up the A vector from a double. Typically does
     * so by direct scaling and offset to an integer lookup, using
     * a fine enough table granularity that there is little error.
     * Alternatively uses linear interpolation.
     * The range of the double is predefined based on knowledge of
     * voltage or conc ranges, and the granularity is specified by
     * the xmin, xmax, and invDx fields.
     */
    double lookupA(double v) const;

    /**
     * lookupB: Look up the B vector from a double, similar to lookupA.
     */
    double lookupB(double v) const;
    //////////////////////////////////////////////////////////
    // DestFinfos
    //////////////////////////////////////////////////////////
    /**
     * Single call to get both A and B values by lookup
     */
    void lookupBoth(double v, double* A, double* B) const;
    /// Set the expression for evaluating alpha
    void setAlpha(const Eref& e, const string expr);
    string getAlpha(const Eref& e) const;
    /// Set the expression for evaluating beta
    void setBeta(const Eref& e, const string expr);
    string getBeta(const Eref& e) const;
    /// Set the expression for evaluating tau
    void setTau(const Eref& e, const string expr);
    string getTau(const Eref& e) const;
    /// Set the expression for evaluating mInfinity
    void setMinfinity(const Eref& e, const string expr);
    string getMinfinity(const Eref& e) const;

    /////////////////////////////////////////////////////////////////
    // Utility funcs
    /////////////////////////////////////////////////////////////////
    static const Cinfo* initCinfo();

private:
    /// Whether the gate is expressed in tau-inf form. If false, it is
    /// alpha-beta form
    bool tauInf_;
    exprtk::symbol_table<double> symTab_;
    exprtk::expression<double> alpha_;
    exprtk::expression<double> beta_;
    exprtk::parser<double> parser_;
    mutable double v_;
    /// Store the user-specified expression strings
    string alphaExpr_;
    string betaExpr_;
};

#endif  // _HHGateF_h

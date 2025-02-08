/**********************************************************************
 ** This program is part of 'MOOSE', the
 ** Messaging Object Oriented Simulation Environment.
 **           Copyright (C) 2003-2007 Upinder S. Bhalla. and NCBS
 ** It is made available under the terms of the
 ** GNU Lesser General Public License version 2.1
 ** See the file COPYING.LIB for the full notice.
 **********************************************************************/

#include "../basecode/header.h"
#include "../basecode/ElementValueFinfo.h"
#include "HHGateF.h"

const Cinfo* HHGateF::initCinfo()
{
    ///////////////////////////////////////////////////////
    // Field definitions.
    ///////////////////////////////////////////////////////
    static ReadOnlyLookupValueFinfo<HHGateF, double, double> A(
        "A",
        "lookupA: Compute the A gate value from a double. "
        "This is done by evaluating the expressions for alpha/beta"
        " or tau/inf.",
        &HHGateF::lookupA);
    static ReadOnlyLookupValueFinfo<HHGateF, double, double> B(
        "B",
        "lookupB: Look up the B gate value from a double."
        "This is done by evaluating the expressions for alpha/beta"
        " or tau/inf.",
        &HHGateF::lookupB);

    static ElementValueFinfo<HHGateF, string> alpha(
        "alpha",
        "Expression for voltage-dependent rates, alpha. "
        "This requires the expression for beta to be defined as well.",
        &HHGateF::setAlpha, &HHGateF::getAlpha);

    static ElementValueFinfo<HHGateF, string> beta(
        "beta",
        "Expression for voltage-dependent rates, beta. "
        "This requires the expression for alpha to be defined as well.",
        &HHGateF::setBeta, &HHGateF::getBeta);

    static ElementValueFinfo<HHGateF, string> tau(
        "tau",
        "Expression for voltage-dependent rates, tau. "
        "This requires the expression for mInfinity to be defined as well.",
        &HHGateF::setTau, &HHGateF::getTau);

    static ElementValueFinfo<HHGateF, string> mInfinity(
        "mInfinity",
        "Expression for voltage-dependent rates, mInfinity. "
        "This requires the expression for tau to be defined as well.",
        &HHGateF::setMinfinity, &HHGateF::getMinfinity);

    ///////////////////////////////////////////////////////
    // DestFinfos
    ///////////////////////////////////////////////////////
    static Finfo* HHGateFFinfos[] = {
        &A,          // ReadOnlyLookupValue
        &B,          // ReadOnlyLookupValue
        &alpha,      // Value
        &beta,       // Value
        &tau,        // Value
        &mInfinity,  // Value
    };

    static string doc[] = {
        "Name",
        "HHGateF",
        "Author",
        "Subhasis Ray, 2025, CHINTA",
        "Description",
        "HHGateF: Gate for Hodkgin-Huxley type channels, equivalent to the "
        "m and h terms on the Na squid channel and the n term on K. "
        "This takes the voltage and state variable from the channel, "
        "computes the new value of the state variable and a scaling, "
        "depending on gate power, for the conductance. As opposed to HHGate, "
        "which uses lookup tables for speed, this evaluates explicit "
        "expressions for accuracy. This is a single variable gate, either "
        "voltage or concentration. So the expression also allows only one "
        "indpendent variable, which is assumed `v`. See the documentation of "
        "``Function`` class for details on the praser.",
    };

    static Dinfo<HHGateF> dinfo;
    static Cinfo HHGateFCinfo("HHGateF", Neutral::initCinfo(), HHGateFFinfos,
                              sizeof(HHGateFFinfos) / sizeof(Finfo*), &dinfo,
                              doc, sizeof(doc) / sizeof(string));

    return &HHGateFCinfo;
}

static const Cinfo* hhGateCinfo = HHGateF::initCinfo();
///////////////////////////////////////////////////
// Core class functions
///////////////////////////////////////////////////
HHGateF::HHGateF() : HHGateBase(0, 0)
{
    cerr << "Warning: HHGateF::HHGateF(): this should never be called" << endl;
}

HHGateF::HHGateF(Id originalChanId, Id originalGateId)
    : HHGateBase(originalChanId, originalGateId)

{
    symTab_.add_variable("v", v_);
    symTab_.add_constants();
    alpha_.register_symbol_table(symTab_);
    beta_.register_symbol_table(symTab_);
}

HHGateF& HHGateF::operator=(const HHGateF& rhs)
{
    // protect from self-assignment.
    if( this == &rhs)
        return *this;

    v_ = rhs.v_;
    symTab_.add_variable("v_", v_);
    symTab_.add_constants();
    alpha_.register_symbol_table(symTab_);
    beta_.register_symbol_table(symTab_);
    alphaExpr_ = rhs.alphaExpr_;
    betaExpr_ = rhs.betaExpr_;
    parser_.compile(alphaExpr_, alpha_);
    parser_.compile(betaExpr_, beta_);
    tauInf_ = rhs.tauInf_;
    return *this;
}

///////////////////////////////////////////////////
// Field function definitions
///////////////////////////////////////////////////

double HHGateF::lookupA(double v) const
{
    // TODO: check for divide by zero?
    v_ = v;
    return tauInf_ ? beta_.value() / alpha_.value() : alpha_.value();
}

double HHGateF::lookupB(double v) const
{
    // TODO: check for divide by zero?
    v_ = v;
    return tauInf_ ? 1.0 / alpha_.value() : alpha_.value() + beta_.value();
}

void HHGateF::lookupBoth(double v, double* A, double* B) const
{
    *A = lookupA(v);
    *B = lookupB(v);
    cerr << "# HHGateF::lookupBoth: v=" << v << ", A=" << *A << ", B="<< *B << endl;
}

void HHGateF::setAlpha(const Eref& e, const string expr)
{
    if(checkOriginal(e.id(), "alpha")) {
        if(!parser_.compile(expr, alpha_)) {
            cerr << "Error: HHGateF::setAlpha: cannot compile expression!\n"
                 << parser_.error() << endl;
            return;
        }
        tauInf_ = false;
        alphaExpr_ = expr;
    }
}

string HHGateF::getAlpha(const Eref& e) const
{
    return tauInf_ ? "" : alphaExpr_;
}

void HHGateF::setBeta(const Eref& e, const string expr)
{
    if(checkOriginal(e.id(), "beta")) {
        if(!parser_.compile(expr, beta_)) {
            cerr << "Error: HHGateF::setBeta: cannot compile expression!\n"
                 << parser_.error() << endl;
            return;
        }
        tauInf_ = false;
        betaExpr_ = expr;
    }
}

string HHGateF::getBeta(const Eref& e) const
{
    return tauInf_ ? "" : betaExpr_;
}

void HHGateF::setTau(const Eref& e, const string expr)
{
    if(checkOriginal(e.id(), "alpha")) {
        if(!parser_.compile(expr, alpha_)) {
            cerr << "Error: HHGateF::setTau: cannot compile expression!\n"
                 << parser_.error() << endl;
            return;
        }
        tauInf_ = true;
        alphaExpr_ = expr;
    }
}

string HHGateF::getTau(const Eref& e) const
{
    return tauInf_ ? alphaExpr_ : "";
}

void HHGateF::setMinfinity(const Eref& e, const string expr)
{
    if(checkOriginal(e.id(), "beta")) {
        if(!parser_.compile(expr, beta_)) {
            cerr << "Error: HHGateF::setMinfinity: cannot compile expression!\n"
                 << parser_.error() << endl;
            return;
        }
        tauInf_ = true;
        betaExpr_ = expr;
    }
}

string HHGateF::getMinfinity(const Eref& e) const
{
    return tauInf_ ? betaExpr_ : "";
}

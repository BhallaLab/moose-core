// Filename: HHChannelF.cpp
// Description: Formula evaluation based HHChannel
// Author: Subhasis Ray
// Created: Fri Jan 24 13:19:07 2025 (+0530)
//

#include "ChanBase.h"
#include "ChanCommon.h"
#include "HHChannelBase.h"
#include "HHGateF.h"
#include "HHChannelF.h"

const Cinfo* HHChannelF::initCinfo() {
    static FieldElementFinfo<HHChannelF, HHGateF> gateX(
        "gateX", "Sets up HHGate X for channel", HHGateF::initCinfo(),
        &HHChannelF::getXgate, &HHChannelF::setNumGates, &HHChannelF::getNumXgates
        // 1
    );
    static FieldElementFinfo<HHChannelF, HHGateF> gateY(
        "gateY", "Sets up HHGateF Y for channel", HHGateF::initCinfo(),
        &HHChannelF::getYgate, &HHChannelF::setNumGates, &HHChannelF::getNumYgates
        // 1
    );
    static FieldElementFinfo<HHChannelF, HHGateF> gateZ(
        "gateZ", "Sets up HHGateF Z for channel", HHGateF::initCinfo(),
        &HHChannelF::getZgate, &HHChannelF::setNumGates, &HHChannelF::getNumZgates
        // 1
    );
    ///////////////////////////////////////////////////////
    static Finfo* HHChannelFFinfos[] = {
        &gateX,  // FieldElement
        &gateY,  // FieldElement
        &gateZ   // FieldElement
    };

    ///////////////////////////////////////////////////////
    static string doc[] = {
        "Name",
        "HHChannelF",
        "Author",
        "Subhasis Ray, 2025, CHINTA",
        "Description",
        "HHChannelF: Hodgkin-Huxley type voltage-gated Ion channel. Unlike "
	"HHChannel, which uses table lookup for speed, this version evaluates "
	"an expression to compute the gate variables for better accuracy.",
    };

    static Dinfo<HHChannelF> dinfo;

    static Cinfo HHChannelFCinfo("HHChannelF", HHChannelBase::initCinfo(),
                                HHChannelFFinfos,
                                sizeof(HHChannelFFinfos) / sizeof(Finfo*),
                                &dinfo, doc, sizeof(doc) / sizeof(string));

    return &HHChannelFCinfo;
}

static const Cinfo* hhChannelCinfo = HHChannelF::initCinfo();
//////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////
// Constructor
///////////////////////////////////////////////////
HHChannelF::HHChannelF() : conc_(0.0), xGate_(0), yGate_(0), zGate_(0) { ; }

HHChannelF::~HHChannelF() {
    ;
}
/**
 * If the gate exists and has only this element for input, then change
 * the gate power.
 * If the gate exists and has multiple parents, then make a new gate,
 * 	set its power.
 * If the gate does not exist, make a new gate, set its power.
 *
 * The function is designed with the idea that if copies of this
 * channel are made, then they all point back to the original HHGateF.
 * (Unless they are cross-node copies).
 * It is only if we subsequently alter the HHGateF of this channel that
 * we need to make our own variant of the HHGateF, or disconnect from
 * an existing one.
 * \todo: May need to convert to handling arrays and Erefs.
 */
// Assuming that the elements are simple elements. Use Eref for
// general case

bool HHChannelF::checkOriginal(Id chanId) const {
    bool isOriginal = true;
    if (xGate_) {
        isOriginal = xGate_->isOriginalChannel(chanId);
    } else if (yGate_) {
        isOriginal = yGate_->isOriginalChannel(chanId);
    } else if (zGate_) {
        isOriginal = zGate_->isOriginalChannel(chanId);
    }
    return isOriginal;
}

void HHChannelF::innerCreateGate(const string& gateName, HHGateF** gatePtr,
                                Id chanId, Id gateId) {
    // Shell* shell = reinterpret_cast< Shell* >( ObjId( Id(), 0 ).data() );
    if (*gatePtr) {
        cout << "Warning: HHChannelF::createGate: '" << gateName
             << "' on Element '" << chanId.path() << "' already present\n";
        return;
    }
    *gatePtr = new HHGateF(chanId, gateId);
}

void HHChannelF::vCreateGate(const Eref& e, string gateType) {
    if (!checkOriginal(e.id())) {
        cout << "Warning: HHChannelF::createGate: Not allowed from copied "
                "channel:\n"
             << e.id().path() << "\n";
        return;
    }

    if (gateType == "X")
        innerCreateGate("xGate", &xGate_, e.id(), Id(e.id().value() + 1));
    else if (gateType == "Y")
        innerCreateGate("yGate", &yGate_, e.id(), Id(e.id().value() + 2));
    else if (gateType == "Z")
        innerCreateGate("zGate", &zGate_, e.id(), Id(e.id().value() + 3));
    else
        cout << "Warning: HHChannelF::createGate: Unknown gate type '"
             << gateType << "'. Ignored\n";
}

void HHChannelF::innerDestroyGate(const string& gateName, HHGateF** gatePtr,
                                 Id chanId) {
    if (*gatePtr == nullptr) {
        cout << "Warning: HHChannelF::destroyGate: '" << gateName
             << "' on Element '" << chanId.path() << "' not present\n";
        return;
    }
    delete (*gatePtr);
    *gatePtr = nullptr;
}

void HHChannelF::destroyGate(const Eref& e, string gateType) {
    if (!checkOriginal(e.id())) {
        cout << "Warning: HHChannelF::destroyGate: Not allowed from copied "
                "channel:\n"
             << e.id().path() << "\n";
        return;
    }

    if (gateType == "X")
        innerDestroyGate("xGate", &xGate_, e.id());
    else if (gateType == "Y")
        innerDestroyGate("yGate", &yGate_, e.id());
    else if (gateType == "Z")
        innerDestroyGate("zGate", &zGate_, e.id());
    else
        cout << "Warning: HHChannelF::destroyGate: Unknown gate type '"
             << gateType << "'. Ignored\n";
}
///////////////////////////////////////////////////
// HHGateF functions
//
// These are breaking the design as the return type is HHGateF for
// HHChannelF but HHGateF2D for HHChannelF2D. Making a common HHGateBase
// also turns out to be problematic as the field element can no longer
// be accessed as an HHGateF or HHGateF2D.
///////////////////////////////////////////////////

HHGateF* HHChannelF::getXgate(unsigned int i) { return xGate_; }

HHGateF* HHChannelF::getYgate(unsigned int i) { return yGate_; }

HHGateF* HHChannelF::getZgate(unsigned int i) { return zGate_; }

void HHChannelF::setNumGates(unsigned int num) { ; }

unsigned int HHChannelF::getNumXgates() const { return xGate_ != nullptr; }

unsigned int HHChannelF::getNumYgates() const { return yGate_ != nullptr; }

unsigned int HHChannelF::getNumZgates() const { return zGate_ != nullptr; }


///////////////////////////////////////////////////
// Dest function definitions
///////////////////////////////////////////////////

void HHChannelF::vProcess(const Eref& e, ProcPtr info) {
    g_ += ChanCommon::vGetGbar(e);
    double A = 0;
    double B = 0;
    if (Xpower_ > 0) {
        xGate_->lookupBoth(Vm_, &A, &B);
        if (instant_ & INSTANT_X)
            X_ = A / B;
        else
            X_ = integrate(X_, info->dt, A, B);
        g_ *= takeXpower_(X_, Xpower_);
    }

    if (Ypower_ > 0) {
        yGate_->lookupBoth(Vm_, &A, &B);
        if (instant_ & INSTANT_Y)
            Y_ = A / B;
        else
            Y_ = integrate(Y_, info->dt, A, B);

        g_ *= takeYpower_(Y_, Ypower_);
    }

    if (Zpower_ > 0) {
        if (useConcentration_)
            zGate_->lookupBoth(conc_, &A, &B);
        else
            zGate_->lookupBoth(Vm_, &A, &B);
        if (instant_ & INSTANT_Z)
            Z_ = A / B;
        else
            Z_ = integrate(Z_, info->dt, A, B);

        g_ *= takeZpower_(Z_, Zpower_);
    }
    ChanCommon::vSetGk(e, g_ * ChanCommon::vGetModulation(e));
    ChanCommon::updateIk();
    // Gk_ = g_;
    // Ik_ = ( Ek_ - Vm_ ) * g_;

    // Send out the relevant channel messages.
    ChanCommon::sendProcessMsgs(e, info);

    g_ = 0.0;
}

/**
 * Here we get the steady-state values for the gate (the 'instant'
 * calculation) as A_/B_.
 */
void HHChannelF::vReinit(const Eref& er, ProcPtr info) {
    g_ = ChanCommon::vGetGbar(er);
    Element* e = er.element();

    double A = 0.0;
    double B = 0.0;
    if (Xpower_ > 0) {
        assert(xGate_);
        xGate_->lookupBoth(Vm_, &A, &B);
        if (B < EPSILON) {
            cout << "Warning: B_ value for " << e->getName()
                 << " is ~0. Check X table\n";
            return;
        }
        if (!xInited_) X_ = A / B;
        g_ *= takeXpower_(X_, Xpower_);
    }

    if (Ypower_ > 0) {
        assert(yGate_);
        yGate_->lookupBoth(Vm_, &A, &B);
        if (B < EPSILON) {
            cout << "Warning: B value for " << e->getName()
                 << " is ~0. Check Y table\n";
            return;
        }
        if (!yInited_) Y_ = A / B;
        g_ *= takeYpower_(Y_, Ypower_);
    }

    if (Zpower_ > 0) {
        assert(zGate_);
        if (useConcentration_)
            zGate_->lookupBoth(conc_, &A, &B);
        else
            zGate_->lookupBoth(Vm_, &A, &B);
        if (B < EPSILON) {
            cout << "Warning: B value for " << e->getName()
                 << " is ~0. Check Z table\n";
            return;
        }
        if (!zInited_) Z_ = A / B;
        g_ *= takeZpower_(Z_, Zpower_);
    }
    ChanCommon::vSetGk(er, g_ * ChanCommon::vGetModulation(er));
    ChanCommon::updateIk();
    // Gk_ = g_;
    // Ik_ = ( Ek_ - Vm_ ) * g_;

    // Send out the relevant channel messages.
    // Same for reinit as for process.
    ChanCommon::sendReinitMsgs(er, info);

    g_ = 0.0;
}

void HHChannelF::vHandleConc(const Eref& e, double conc) { conc_ = conc; }

//
// HHChannelF.cpp ends here

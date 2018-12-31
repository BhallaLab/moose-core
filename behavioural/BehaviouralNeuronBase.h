/***
 *    Description:  Base class for BehaviouralNeuron.
 *
 *        Created:  2018-12-28

 *         Author:  Dilawar Singh <dilawars@ncbs.res.in>
 *   Organization:  NCBS Bangalore
 *        License:  See LICENSE file.
 */

#ifndef BEHAVIOURAL_NEURON_BASE_H
#define BEHAVIOURAL_NEURON_BASE_H

#include "BehaviouralSystem.h"
#include "../external/muparser/include/muParser.h"

namespace moose {

class BehaviouralNeuronBase : public Compartment, public BehaviouralSystem

{
public:
    BehaviouralNeuronBase();
    virtual ~BehaviouralNeuronBase();

    // Value Field access function definitions.
    void setThresh( const Eref& e,  double val );
    double getThresh( const Eref& e  ) const;

    // Get-Set variables.
    void setODEVariables( const Eref& e, const vector<string> vars);
    vector<string> getODEVariables( const Eref& e ) const;

    void setODEStates( const Eref& e, const vector<string> ss);
    vector<string> getODEStates( const Eref& e ) const;

    // Get-Set equations.
    void setEquations(const Eref& e, const vector<string> eqs);
    vector<string> getEquations(const Eref& e) const;

    // Build system and setup a ODE system.
    void setupParser(mu::Parser& p);
    void buildSystem();

    void setVReset( const Eref& e,  double val );

    double getVReset( const Eref& e  ) const;
    void setRefractoryPeriod( const Eref& e,  double val );

    double getRefractoryPeriod( const Eref& e  ) const;
    double getLastEventTime( const Eref& e  ) const;
    bool hasFired( const Eref& e ) const;

    // Dest function definitions.
    /**
     * The process function does the object updating and sends out
     * messages to channels, nernsts, and so on.
     */
    virtual void vProcess( const Eref& e, ProcPtr p ) = 0;

    /**
     * The reinit function reinitializes all fields.
     */
    virtual void vReinit( const Eref& e, ProcPtr p ) = 0;

    /**
     * activation handles information coming from the SynHandler
     * to the behaviouralNeuron.
     */
    void activation( double val );

    /// Message src for outgoing spikes.
    static SrcFinfo1< double >* spikeOut();

    /**
     * Initializes the class info.
     */
    static const Cinfo* initCinfo();

protected:
    double threshold_;
    double vReset_;
    double activation_;
    double refractT_;
    double lastEvent_;
    bool fired_;
    bool isBuilt_;

    // variables
    vector<string> variables_;
    vector<string> states_;

    // map of y' and its expression parser.
    map<string, mu::Parser> odeMap_;
    map<string, mu::Parser> eqMap_;

    // expression in lhs=rhs form. 
    vector<string> eqs_; 

    // Map of variables. To be used in parser. Initialized in constructors. It
    // keep pointer to data.
    map<string, double*> vals_;
    
};

} // namespace

#endif // _BEHAVIOURAL_NEURON_BASE_H

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

namespace moose
{

class BehaviouralNeuronBase
{
public:
    BehaviouralNeuronBase();
    virtual ~BehaviouralNeuronBase();

    // Value Field access function definitions.
    void setThresh( const Eref& e,  double val );
    double getThresh( const Eref& e  ) const;

    void setODEVariables( const Eref& e, vector<string> vars);
    vector<string> getODEVariables( const Eref& e ) const;

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
    static SrcFinfo1< double >* VmOut();

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
    double Vm_;
    double sumInject_;

    // variables
    vector<string> variables_;
    vector<string> states_;

};

} // namespace

#endif // _BEHAVIOURAL_NEURON_BASE_H

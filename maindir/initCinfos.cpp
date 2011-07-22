#include "header.h"
#include "moose.h"

extern const Cinfo* initAdaptorCinfo();
extern const Cinfo* initAscFileCinfo();
//extern const Cinfo* initAtestCinfo();
//extern const Cinfo* initAverageCinfo();
extern const Cinfo* initBinSynchanCinfo();
extern const Cinfo* initBinomialRngCinfo();
extern const Cinfo* initCaConcCinfo();
extern const Cinfo* initCellCinfo();
extern const Cinfo* initClassCinfo();
extern const Cinfo* initClockJobCinfo();
extern const Cinfo* initCompartmentCinfo();
extern const Cinfo* initCylPanelCinfo();
extern const Cinfo* initDifShellCinfo();
extern const Cinfo* initDiffAmpCinfo();
extern const Cinfo* initDiskPanelCinfo();
extern const Cinfo* initEnzymeCinfo();
extern const Cinfo* initExponentialRngCinfo();
extern const Cinfo* initGammaRngCinfo();
extern const Cinfo* initGeometryCinfo();
extern const Cinfo* initGslIntegratorCinfo();
extern const Cinfo* initGssaStoichCinfo();
extern const Cinfo* initLeakageCinfo();
extern const Cinfo* initHHChannelCinfo();
extern const Cinfo* initHHChannel2DCinfo();
extern const Cinfo* initHHGateCinfo();
extern const Cinfo* initHHGate2DCinfo();
extern const Cinfo* initHSolveCinfo();
extern const Cinfo* initHSolveHubCinfo();
extern const Cinfo* initHemispherePanelCinfo();
extern const Cinfo* initGHKCinfo();
extern const Cinfo* initIntFireCinfo();
extern const Cinfo* initCalculatorCinfo();
#ifdef USE_MUSIC
extern const Cinfo* initMusicCinfo();
extern const Cinfo* initInputEventChannelCinfo();
extern const Cinfo* initInputEventPortCinfo();
extern const Cinfo* initOutputEventChannelCinfo();
extern const Cinfo* initOutputEventPortCinfo();
#endif
extern const Cinfo* initInterSolverFluxCinfo();
extern const Cinfo* initInterpolCinfo();
extern const Cinfo* initInterpol2DCinfo();
extern const Cinfo* initKinComptCinfo();
extern const Cinfo* initKinPlaceHolderCinfo();
extern const Cinfo* initKineticHubCinfo();
extern const Cinfo* initKineticManagerCinfo();
extern const Cinfo* initKintegratorCinfo();
extern const Cinfo* initMathFuncCinfo();
extern const Cinfo* initMg_blockCinfo();
extern const Cinfo* initMoleculeCinfo();
extern const Cinfo* initNernstCinfo();
extern const Cinfo* initNeutralCinfo();
extern const Cinfo* initNormalRngCinfo();
extern const Cinfo* initPIDControllerCinfo();
extern const Cinfo* initPanelCinfo();
extern const Cinfo* initIzhikevichNrnCinfo();
#ifdef USE_MPI
//extern const Cinfo* initParGenesisParserCinfo();
extern const Cinfo* initParTickCinfo();
extern const Cinfo* initPostMasterCinfo();
#else
extern const Cinfo* initGenesisParserCinfo();
extern const Cinfo* initTickCinfo();
#endif
#ifdef USE_SMOLDYN
extern const Cinfo* initParticleCinfo();
extern const Cinfo* initSmoldynHubCinfo();
#endif
extern const Cinfo* initPoissonRngCinfo();
extern const Cinfo* initPulseGenCinfo();
extern const Cinfo * initEfieldCinfo();
#ifdef PYMOOSE
extern const Cinfo* initPyMooseContextCinfo();
#endif
extern const Cinfo* initRCCinfo();
extern const Cinfo* initRandGeneratorCinfo();
extern const Cinfo* initRandomSpikeCinfo();
extern const Cinfo* initReactionCinfo();
extern const Cinfo* initRectPanelCinfo();
extern const Cinfo* initShellCinfo();
extern const Cinfo* initSigNeurCinfo();
extern const Cinfo* initSpherePanelCinfo();
extern const Cinfo* initSpikeGenCinfo();
extern const Cinfo* initStochSynchanCinfo();
extern const Cinfo* initStoichCinfo();
extern const Cinfo* initSurfaceCinfo();
extern const Cinfo* initSymCompartmentCinfo();
extern const Cinfo* initSynChanCinfo();
extern const Cinfo* initNMDAChanCinfo();
extern const Cinfo* initSTPSynChanCinfo();
extern const Cinfo* initSTPNMDAChanCinfo();
extern const Cinfo* initTableCinfo();
extern const Cinfo* initTauPumpCinfo();
#ifdef USE_GL
extern const Cinfo* initGLcellCinfo();
extern const Cinfo* initGLviewCinfo();
#endif
extern const Cinfo* initTimeTableCinfo();
extern const Cinfo* initTriPanelCinfo();
extern const Cinfo* initUniformRngCinfo();
extern const Cinfo* initSteadyStateCinfo();
extern const Cinfo* initscript_outCinfo();

const Cinfo ** initCinfos(){
    static const Cinfo * cinfoList[] = {
        initAdaptorCinfo(),
        initAscFileCinfo(),
        //     initAtestCinfo(),
        //     initAverageCinfo(),
        initBinSynchanCinfo(),
        initBinomialRngCinfo(),
        initCaConcCinfo(),
        initCellCinfo(),
        initClassCinfo(),
        initClockJobCinfo(),
        initCompartmentCinfo(),
        initCylPanelCinfo(),
        initDifShellCinfo(),
        initDiffAmpCinfo(),
        initDiskPanelCinfo(),
        initEnzymeCinfo(),
        initExponentialRngCinfo(),
        initGammaRngCinfo(),
        initGeometryCinfo(),
        initGHKCinfo(),
        initIntFireCinfo(),
        initCalculatorCinfo(),

#ifdef USE_GSL
        initGslIntegratorCinfo(),
        initSteadyStateCinfo(),
#endif
        initGssaStoichCinfo(),
        initInterpolCinfo(),
        initInterpol2DCinfo(),
        initHHGateCinfo(),
        initHHGate2DCinfo(),
        initHHChannelCinfo(),
        initHHChannel2DCinfo(),
        initLeakageCinfo(),
        initHSolveCinfo(),
        initHSolveHubCinfo(),
        initHemispherePanelCinfo(),
#ifdef USE_MUSIC
        initInputEventChannelCinfo(),
        initInputEventPortCinfo(),
        initMusicCinfo(),
        initOutputEventChannelCinfo(),
        initOutputEventPortCinfo(),
#endif
        initInterSolverFluxCinfo(),
        initKinComptCinfo(),
        initKinPlaceHolderCinfo(),
        initKineticHubCinfo(),
        initKineticManagerCinfo(),
        initKintegratorCinfo(),
        initMathFuncCinfo(),
        initMg_blockCinfo(),
        initMoleculeCinfo(),
        initNernstCinfo(),
        initNeutralCinfo(),
        initNormalRngCinfo(),
        initPIDControllerCinfo(),
        initPanelCinfo(),
        initIzhikevichNrnCinfo(),
#ifdef USE_MPI
        //     initParGenesisParserCinfo(),
        initParTickCinfo(),
        initPostMasterCinfo(),
#else
        initGenesisParserCinfo(),
        initTickCinfo(),
#endif
#ifdef USE_SMOLDYN
        initSmoldynHubCinfo(),
        initParticleCinfo(),
#endif
        initPoissonRngCinfo(),
        initPulseGenCinfo(),
        initEfieldCinfo(),
#ifdef PYMOOSE
        initPyMooseContextCinfo(),
#endif
        initRCCinfo(),
        initRandGeneratorCinfo(),
        initRandomSpikeCinfo(),
        initReactionCinfo(),
        initRectPanelCinfo(),
        initShellCinfo(),
        initSigNeurCinfo(),
        initSpherePanelCinfo(),
        initSpikeGenCinfo(),
        initStochSynchanCinfo(),
        initStoichCinfo(),
        initSurfaceCinfo(),
        initSymCompartmentCinfo(),
        initSynChanCinfo(),
        initNMDAChanCinfo(),
        initSTPSynChanCinfo(),
        initSTPNMDAChanCinfo(),
        initTableCinfo(),
        initTauPumpCinfo(),
#ifdef USE_GL
        initGLcellCinfo(),
        initGLviewCinfo(),
#endif
        initTimeTableCinfo(),
        initTriPanelCinfo(),
        initUniformRngCinfo(),
        initscript_outCinfo(),
    };

    return cinfoList;
    
    
}

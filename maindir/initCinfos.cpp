#include "header.h"
#include "moose.h"

extern const Cinfo* initAdaptorCinfo();
extern const Cinfo* initAscFileCinfo();
extern const Cinfo* initAtestCinfo();
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
extern const Cinfo* initGenesisParserCinfo();
extern const Cinfo* initGeometryCinfo();
extern const Cinfo* initGslIntegratorCinfo();
extern const Cinfo* initGssaStoichCinfo();
extern const Cinfo* initHHChannelCinfo();
extern const Cinfo* initHHGateCinfo();
extern const Cinfo* initHSolveCinfo();
extern const Cinfo* initHSolveHubCinfo();
extern const Cinfo* initHemispherePanelCinfo();
#ifdef USE_MUSIC
extern const Cinfo* initInputEventChannelCinfo();
extern const Cinfo* initInputEventPortCinfo();
#endif
extern const Cinfo* initInterSolverFluxCinfo();
extern const Cinfo* initInterpolCinfo();
extern const Cinfo* initKinComptCinfo();
extern const Cinfo* initKinPlaceHolderCinfo();
extern const Cinfo* initKineticHubCinfo();
extern const Cinfo* initKineticManagerCinfo();
extern const Cinfo* initKintegratorCinfo();
extern const Cinfo* initMathFuncCinfo();
extern const Cinfo* initMg_blockCinfo();
extern const Cinfo* initMoleculeCinfo();
#ifdef USE_MUSIC
extern const Cinfo* initMusicCinfo();
#endif
extern const Cinfo* initNernstCinfo();
extern const Cinfo* initNeutralCinfo();
extern const Cinfo* initNormalRngCinfo();
#ifdef USE_MUSIC
extern const Cinfo* initOutputEventChannelCinfo();
extern const Cinfo* initOutputEventPortCinfo();
#endif
extern const Cinfo* initPIDControllerCinfo();
extern const Cinfo* initPanelCinfo();
#ifdef USE_MPI
extern const Cinfo* initParGenesisParserCinfo();
extern const Cinfo* initParTickCinfo();
#endif
#ifdef USE_SMOLDYN
extern const Cinfo* initParticleCinfo();
#endif
extern const Cinfo* initPoissonRngCinfo();
extern const Cinfo* initPostMasterCinfo();
extern const Cinfo* initPulseGenCinfo();
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
#ifdef USE_SMOLDYN
extern const Cinfo* initSmoldynHubCinfo();
#endif
extern const Cinfo* initSpherePanelCinfo();
extern const Cinfo* initSpikeGenCinfo();
extern const Cinfo* initStochSynchanCinfo();
extern const Cinfo* initStoichCinfo();
extern const Cinfo* initSurfaceCinfo();
extern const Cinfo* initSymCompartmentCinfo();
extern const Cinfo* initSynChanCinfo();
extern const Cinfo* initTableCinfo();
extern const Cinfo* initTauPumpCinfo();
extern const Cinfo* initTickCinfo();
extern const Cinfo* initTimeTableCinfo();
extern const Cinfo* initTriPanelCinfo();
extern const Cinfo* initUniformRngCinfo();
extern const Cinfo* initscript_outCinfo();

void initCinfos(){
    static const Cinfo* AdaptorCinfo = initAdaptorCinfo();
    static const Cinfo* AscFileCinfo = initAscFileCinfo();
    static const Cinfo* AtestCinfo = initAtestCinfo();
    //    static const Cinfo* AverageCinfo = initAverageCinfo();
    static const Cinfo* BinSynchanCinfo = initBinSynchanCinfo();
    static const Cinfo* BinomialRngCinfo = initBinomialRngCinfo();
    static const Cinfo* CaConcCinfo = initCaConcCinfo();
    static const Cinfo* CellCinfo = initCellCinfo();
    static const Cinfo* ClassCinfo = initClassCinfo();
    static const Cinfo* ClockJobCinfo = initClockJobCinfo();
    static const Cinfo* CompartmentCinfo = initCompartmentCinfo();
    static const Cinfo* CylPanelCinfo = initCylPanelCinfo();
    static const Cinfo* DifShellCinfo = initDifShellCinfo();
    static const Cinfo* DiffAmpCinfo = initDiffAmpCinfo();
    static const Cinfo* DiskPanelCinfo = initDiskPanelCinfo();
    static const Cinfo* EnzymeCinfo = initEnzymeCinfo();
    static const Cinfo* ExponentialRngCinfo = initExponentialRngCinfo();
    static const Cinfo* GammaRngCinfo = initGammaRngCinfo();
    static const Cinfo* GenesisParserCinfo = initGenesisParserCinfo();
    static const Cinfo* GeometryCinfo = initGeometryCinfo();
#ifdef USE_GSL
    static const Cinfo* GslIntegratorCinfo = initGslIntegratorCinfo();
#endif
    static const Cinfo* GssaStoichCinfo = initGssaStoichCinfo();
    static const Cinfo* HHChannelCinfo = initHHChannelCinfo();
    static const Cinfo* HHGateCinfo = initHHGateCinfo();
    static const Cinfo* HSolveCinfo = initHSolveCinfo();
    static const Cinfo* HSolveHubCinfo = initHSolveHubCinfo();
    static const Cinfo* HemispherePanelCinfo = initHemispherePanelCinfo();
#ifdef USE_MUSIC
    static const Cinfo* InputEventChannelCinfo = initInputEventChannelCinfo();
    static const Cinfo* InputEventPortCinfo = initInputEventPortCinfo();
#endif
    static const Cinfo* InterSolverFluxCinfo = initInterSolverFluxCinfo();
    static const Cinfo* InterpolCinfo = initInterpolCinfo();
    static const Cinfo* KinComptCinfo = initKinComptCinfo();
    static const Cinfo* KinPlaceHolderCinfo = initKinPlaceHolderCinfo();
    static const Cinfo* KineticHubCinfo = initKineticHubCinfo();
    static const Cinfo* KineticManagerCinfo = initKineticManagerCinfo();
    static const Cinfo* KintegratorCinfo = initKintegratorCinfo();
    static const Cinfo* MathFuncCinfo = initMathFuncCinfo();
    static const Cinfo* Mg_blockCinfo = initMg_blockCinfo();
    static const Cinfo* MoleculeCinfo = initMoleculeCinfo();
#ifdef USE_MUSIC
    static const Cinfo* MusicCinfo = initMusicCinfo();
#endif
    static const Cinfo* NernstCinfo = initNernstCinfo();
    static const Cinfo* NeutralCinfo = initNeutralCinfo();
    static const Cinfo* NormalRngCinfo = initNormalRngCinfo();
#ifdef USE_MUSIC
    static const Cinfo* OutputEventChannelCinfo = initOutputEventChannelCinfo();
    static const Cinfo* OutputEventPortCinfo = initOutputEventPortCinfo();
#endif
    static const Cinfo* PIDControllerCinfo = initPIDControllerCinfo();
    static const Cinfo* PanelCinfo = initPanelCinfo();
#ifdef USE_MPI
    static const Cinfo* ParGenesisParserCinfo = initParGenesisParserCinfo();
    static const Cinfo* ParTickCinfo = initParTickCinfo();
#endif
#ifdef USE_SMOLDYN
    static const Cinfo* ParticleCinfo = initParticleCinfo();
#endif
    static const Cinfo* PoissonRngCinfo = initPoissonRngCinfo();
#ifdef USE_MPI
    static const Cinfo* PostMasterCinfo = initPostMasterCinfo();
#endif
    static const Cinfo* PulseGenCinfo = initPulseGenCinfo();
#ifdef PYMOOSE
    static const Cinfo* PyMooseContextCinfo = initPyMooseContextCinfo();
#endif
    static const Cinfo* RCCinfo = initRCCinfo();
    static const Cinfo* RandGeneratorCinfo = initRandGeneratorCinfo();
    static const Cinfo* RandomSpikeCinfo = initRandomSpikeCinfo();
    static const Cinfo* ReactionCinfo = initReactionCinfo();
    static const Cinfo* RectPanelCinfo = initRectPanelCinfo();
    static const Cinfo* ShellCinfo = initShellCinfo();
    static const Cinfo* SigNeurCinfo = initSigNeurCinfo();
#ifdef USE_SMOLDYN
    static const Cinfo* SmoldynHubCinfo = initSmoldynHubCinfo();
#endif
    static const Cinfo* SpherePanelCinfo = initSpherePanelCinfo();
    static const Cinfo* SpikeGenCinfo = initSpikeGenCinfo();
    static const Cinfo* StochSynchanCinfo = initStochSynchanCinfo();
    static const Cinfo* StoichCinfo = initStoichCinfo();
    static const Cinfo* SurfaceCinfo = initSurfaceCinfo();
    static const Cinfo* SymCompartmentCinfo = initSymCompartmentCinfo();
    static const Cinfo* SynChanCinfo = initSynChanCinfo();
    static const Cinfo* TableCinfo = initTableCinfo();
    static const Cinfo* TauPumpCinfo = initTauPumpCinfo();
    static const Cinfo* TickCinfo = initTickCinfo();
    static const Cinfo* TimeTableCinfo = initTimeTableCinfo();
    static const Cinfo* TriPanelCinfo = initTriPanelCinfo();
    static const Cinfo* UniformRngCinfo = initUniformRngCinfo();
    static const Cinfo* script_outCinfo = initscript_outCinfo();
}

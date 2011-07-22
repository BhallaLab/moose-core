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

void initCinfos(){
    static const Cinfo* AdaptorCinfo = initAdaptorCinfo();
    static const Cinfo* AscFileCinfo = initAscFileCinfo();
//    static const Cinfo* AtestCinfo = initAtestCinfo();
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
    static const Cinfo* GeometryCinfo = initGeometryCinfo();
    static const Cinfo* GHKCinfo = initGHKCinfo();
    static const Cinfo* IntFireCinfo = initIntFireCinfo();
    static const Cinfo* CalculatorCinfo = initCalculatorCinfo();

#ifdef USE_GSL
    static const Cinfo* GslIntegratorCinfo = initGslIntegratorCinfo();
    static const Cinfo* SteadyStateCinfo = initSteadyStateCinfo();
#endif
    static const Cinfo* GssaStoichCinfo = initGssaStoichCinfo();
    static const Cinfo* InterpolCinfo = initInterpolCinfo();
    static const Cinfo* Interpol2DCinfo = initInterpol2DCinfo();
    static const Cinfo* HHGateCinfo = initHHGateCinfo();
    static const Cinfo* HHGate2DCinfo = initHHGate2DCinfo();
    static const Cinfo* HHChannelCinfo = initHHChannelCinfo();
    static const Cinfo* HHChannel2DCinfo = initHHChannel2DCinfo();
    static const Cinfo* LeakageCinfo = initLeakageCinfo();
    static const Cinfo* HSolveCinfo = initHSolveCinfo();
    static const Cinfo* HSolveHubCinfo = initHSolveHubCinfo();
    static const Cinfo* HemispherePanelCinfo = initHemispherePanelCinfo();
#ifdef USE_MUSIC
    static const Cinfo* InputEventChannelCinfo = initInputEventChannelCinfo();
    static const Cinfo* InputEventPortCinfo = initInputEventPortCinfo();
    static const Cinfo* MusicCinfo = initMusicCinfo();
    static const Cinfo* OutputEventChannelCinfo = initOutputEventChannelCinfo();
    static const Cinfo* OutputEventPortCinfo = initOutputEventPortCinfo();
#endif
    static const Cinfo* InterSolverFluxCinfo = initInterSolverFluxCinfo();
    static const Cinfo* KinComptCinfo = initKinComptCinfo();
    static const Cinfo* KinPlaceHolderCinfo = initKinPlaceHolderCinfo();
    static const Cinfo* KineticHubCinfo = initKineticHubCinfo();
    static const Cinfo* KineticManagerCinfo = initKineticManagerCinfo();
    static const Cinfo* KintegratorCinfo = initKintegratorCinfo();
    static const Cinfo* MathFuncCinfo = initMathFuncCinfo();
    static const Cinfo* Mg_blockCinfo = initMg_blockCinfo();
    static const Cinfo* MoleculeCinfo = initMoleculeCinfo();
    static const Cinfo* NernstCinfo = initNernstCinfo();
    static const Cinfo* NeutralCinfo = initNeutralCinfo();
	static const Cinfo* NormalRngCinfo = initNormalRngCinfo();
    static const Cinfo* PIDControllerCinfo = initPIDControllerCinfo();
    static const Cinfo* PanelCinfo = initPanelCinfo();
	static const Cinfo* IzhikevichNrn = initIzhikevichNrnCinfo();
#ifdef USE_MPI
//    static const Cinfo* ParGenesisParserCinfo = initParGenesisParserCinfo();
    static const Cinfo* ParTickCinfo = initParTickCinfo();
    static const Cinfo* PostMasterCinfo = initPostMasterCinfo();
#else
    static const Cinfo* GenesisParserCinfo = initGenesisParserCinfo();
    static const Cinfo* TickCinfo = initTickCinfo();
#endif
#ifdef USE_SMOLDYN
    static const Cinfo* SmoldynHubCinfo = initSmoldynHubCinfo();
    static const Cinfo* ParticleCinfo = initParticleCinfo();
#endif
    static const Cinfo* PoissonRngCinfo = initPoissonRngCinfo();
    static const Cinfo* PulseGenCinfo = initPulseGenCinfo();
    static const Cinfo* EfieldCinfo = initEfieldCinfo();
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
    static const Cinfo* SpherePanelCinfo = initSpherePanelCinfo();
    static const Cinfo* SpikeGenCinfo = initSpikeGenCinfo();
    static const Cinfo* StochSynchanCinfo = initStochSynchanCinfo();
    static const Cinfo* StoichCinfo = initStoichCinfo();
    static const Cinfo* SurfaceCinfo = initSurfaceCinfo();
    static const Cinfo* SymCompartmentCinfo = initSymCompartmentCinfo();
    static const Cinfo* SynChanCinfo = initSynChanCinfo();
    static const Cinfo* NMDAChanCinfo = initNMDAChanCinfo();
    static const Cinfo* STPSynChanCinfo = initSTPSynChanCinfo();
    static const Cinfo* STPNMDAChanCinfo = initSTPNMDAChanCinfo();
    static const Cinfo* TableCinfo = initTableCinfo();
    static const Cinfo* TauPumpCinfo = initTauPumpCinfo();
#ifdef USE_GL
    static const Cinfo* GLcellCinfo = initGLcellCinfo();
    static const Cinfo* GLviewCinfo = initGLviewCinfo();
#endif
    static const Cinfo* TimeTableCinfo = initTimeTableCinfo();
    static const Cinfo* TriPanelCinfo = initTriPanelCinfo();
    static const Cinfo* UniformRngCinfo = initUniformRngCinfo();
    static const Cinfo* script_outCinfo = initscript_outCinfo();
    
    
}

# -*- coding: utf-8 -*-
import moose
from fixXreacs import fixXreacs

def positionCompt( compt ):
    i = 0
    while (i != len(compt)-1):
        #print "PositionCompt ", compt[i+1],compt[i+1].volume, compt[i], compt[i].volume
        compt[i+1].x1 += compt[i].x1
        compt[i+1].x0 += compt[i].x1
        i += 1

def mooseDeleteChemSolver(modelRoot):
    """Delete solvers from Chemical Compartment    """

    compts = moose.wildcardFind(modelRoot + '/##[ISA=ChemCompt]')
    for compt in compts:
        if moose.exists(compt.path + '/stoich'):
            st = moose.element(compt.path + '/stoich')
            st_ksolve = st.ksolve
            st_dsolve = st.dsolve
    
            moose.delete(st)
    
            if moose.exists((st_ksolve).path):
                print("KSolver is deleted for modelpath %s " % st_ksolve)
                moose.delete(st_ksolve)
                
            if moose.exists((st_dsolve).path) and st_dsolve.path != '/':
                print("DSolver is deleted for modelpath %s " % st_dsolve)
                moose.delete(st_dsolve)
                
def stdSolvertype(solverName):
    if solverName.lower() in ["gssa","gillespie","stochastic","gsolve"]:
        return "gssa"
    elif solverName.lower() in ["gsl","runge kutta","deterministic","ksolve","rungekutta","rk5","rkf","rk"]:
        return "gsl"
    elif solverName.lower() in ["ee","exponential euler","exponentialeuler","neutral"]:
        return "ee"
    return "ee"

def mooseAddChemSolver(modelRoot, solver):
    """    Add the solvers to Chemical compartment     """

    compt = moose.wildcardFind(modelRoot + '/##[ISA=ChemCompt]')
    if not compt:
        return ("Atleast one compartment is required ")
    elif ( len(compt) > 3 ):
        return ("Warning: setSolverOnCompt Cannot handle " ,  len(compt) , " chemical compartments\n")

    else:
        comptinfo = moose.Annotator(moose.element(compt[0]).path + '/info')
        
        previousSolver = stdSolvertype(comptinfo.solver)
        currentSolver = stdSolvertype(solver)

        if previousSolver != currentSolver:
            comptinfo.solver = currentSolver
            if (moose.exists(compt[0].path + '/stoich')):
                # "A: and stoich exists then delete the stoich add solver"
                mooseDeleteChemSolver(modelRoot)
            setCompartmentSolver(modelRoot, currentSolver)
            return True
        else:
            if not moose.exists(compt[0].path + '/stoich'):
                # " stoich exist, doing nothing"
                setCompartmentSolver(modelRoot, currentSolver)
                return True
    return False


def setCompartmentSolver(modelRoot, solver):
    comptlist = dict((c.volume, c) for c in moose.wildcardFind(modelRoot + '/##[ISA=ChemCompt]'))
    vollist = sorted(comptlist.keys())
    compts = [comptlist[key] for key in vollist]
    #compts = [key for key, value in sorted(comptlist.items(), key=lambda (k,v): (v,k))] 
    
    if solver != 'ee':
        if (len(compts) >1 ):
            positionCompt(compts)
            fixXreacs( modelRoot )
            
    vollist = sorted(comptlist.keys())
    compts = [comptlist[key] for key in vollist]
    #compts = [key for key, value in sorted(comptlist.items(), key=lambda (k,v): (v,k))] 

    for compt in compts:
        if solver != 'ee':
            if (solver == 'gsl'):
                ksolve = moose.Ksolve(compt.path + '/ksolve')
            if (solver == 'gssa') :
                ksolve = moose.Gsolve(compt.path + '/gsolve')
            
            if (len(compts) > 1):
                dsolve = moose.Dsolve(compt.path+'/dsolve')
            
            stoich = moose.Stoich(compt.path + '/stoich')
            stoich.ksolve = ksolve
            if (len(compts) > 1):
                stoich.dsolve = dsolve
            
            stoich.compartment = compt
            stoich.path = compt.path + "/##"
            

    dsolveList = moose.wildcardFind(modelRoot+'/##[ISA=Dsolve]')
    i = 0
    while(i < len(dsolveList)-1):
        dsolveList[i+1].buildMeshJunctions(dsolveList[i])
        i += 1
    
    print( " Solver is added to model path %s with %s solver" % (modelRoot,solver) )

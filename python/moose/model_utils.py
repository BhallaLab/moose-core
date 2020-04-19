# -*- coding: utf-8 -*-
# Utilties for loading NML and SBML models.
# Authored and maintained by Harsha Rani

from __future__ import absolute_import, print_function, division

import os
import moose._moose as _moose
import moose.utils as mu

import logging

logger_ = logging.getLogger("moose.model")

# sbml import.
sbmlImport_, sbmlError_ = True, ""
try:
    import moose.SBML.readSBML as _readSBML
    import moose.SBML.writeSBML as _writeSBML
except Exception as e:
    sbmlImport_ = False
    sbmlError_ = str(e)

# NeuroML2 import.
nml2Import_, nml2ImportError_ = True, ""
try:
    import moose.neuroml2 as _neuroml2
except Exception as e:
    nml2Import_ = False
    nml2ImportError_ = str(e)

chemImport_, chemError_ = True, ""
try:
    import moose.chemUtil as _chemUtil
except Exception as e:
    chemImport_ = False
    chemError_ = str(e)

kkitImport_, kkitImportErr_ = True, ""
try:
    import moose.genesis.writeKkit as _writeKkit
except ImportError as e:
    kkitImport_ = False
    kkitImportErr_ = str(e)

mergechemImport_, mergechemError_ = True, ""
try:
    import moose.chemMerge as _chemMerge
except Exception as e:
    mergechemImport_ = False
    mergechemError_ = str(e)

# SBML related functions.
def mooseReadSBML(filepath, loadpath, solver="ee", validate=True):
    """Load SBML model (inner helper function for readSBML)."""
    global sbmlImport_, sbmlError_
    if not sbmlImport_:
        raise ImportError(
            "SBML support could not be loaded because of '%s'" % sbmlError_
        )

    modelpath = _readSBML.mooseReadSBML(filepath, loadpath, solver, validate)
    sc = solver.lower().replace(" ", "")
    if sc in ["gssa", "gillespie", "stochastic", "gsolve"]:
        method = "gssa"
    elif sc in ["gsl", "deterministic", "rungekutta", "rk5", "rk"]:
        method = "gsl"
    else:
        method = "ee"

    if method != "ee":
        _chemUtil.add_Delete_ChemicalSolver.mooseAddChemSolver(
            modelpath[0].path, method
        )
    return modelpath


def mooseWriteSBML(modelpath, filepath, sceneitems={}):
    """Writes loaded model under modelpath to a file in SBML format. 
    (helper function for writeSBML).
    """
    global sbmlImport_, sbmlError_
    if not sbmlImport_:
        raise ImportError(
            "SBML support could not be loaded because of '%s'" % sbmlError_
        )
    return _writeSBML.mooseWriteSBML(modelpath, filepath, sceneitems)


def mooseWriteKkit(modelpath, filepath, sceneitems={}):
    """Writes  loded model under modelpath to a file in Kkit format. (inner
    helper function for moose.writeKkit)"""
    global kkitImport_, kkitImportErr_
    if not kkitImport_:
        raise ImportError(
            "Kkit support could not be enabled becase %s." % kkitImportErr_
        )
    return _writeKkit.mooseWriteKkit(modelpath, filepath, sceneitems)


def mooseDeleteChemSolver(modelpath):
    """deletes solver on all the compartment and its children. (helper function
    for moose.deleteChemSolver)

    Notes
    -----
    This is neccesary while created a new moose object on a pre-existing modelpath,
    this should be followed by mooseAddChemSolver for add solvers on to compartment
    to simulate else default is Exponential Euler (ee)
    """
    if not chemImport_:
        raise ImportError("Failed to load this utility because of %s" % chemError_)
    return _chemUtil.add_Delete_ChemicalSolver.mooseDeleteChemSolver(modelpath)


def mooseAddChemSolver(modelpath, solver):
    """mooseAddChemSolver (helper function for addChemSolver)"""
    if not chemImport_:
        raise ImportError("Could not load chemUtil because %s" % chemError_)
    return _chemUtil.add_Delete_ChemicalSolver.mooseAddChemSolver(modelpath, solver)


def mooseMergeChemModel(src, des):
    """Merges two chemical model.

    File or filepath can be passed source is merged to destination
    """
    global mergechemImport_, mergechemError_
    if not mergechemImport_:
        raise ImportError("Failed to load this utility because of %s" % mergechemError_)
    return _chemMerge.merge.mergeChemModel(src, des)


# NML2 reader and writer function.
def mooseReadNML2(modelpath, verbose=False):
    """Read NeuroML model (version 2) and return reader object.
    """
    global nml2Import_, nml2ImportError_
    if not nml2Import_:
        raise RuntimeError("Could not load NML2 support:\n %s" % nml2ImportError_)

    reader = _neuroml2.NML2Reader(verbose=verbose)
    reader.read(modelpath)
    return reader


def mooseWriteNML2(outfile):
    raise NotImplementedError("Writing to NML2 is not supported yet")


def mooseReadKkitGenesis(filename, modelpath, solverclass="gsl"):
    """helper function. Called by `loadModel` function.

    Only for genesis and cspace.
    """

    if not os.path.isfile(os.path.realpath(filename)):
        raise FileNotFoundError("Model file '%s' found." % filename)

    ext = os.path.splitext(filename)[1]
    sc = solverclass.lower().replace(" ", "")

    if ext in [".swc", ".p"]:
        return _moose.loadModelInternal(filename, modelpath, solverclass)

    if ext in [".g", ".cspace"]:
        # only if genesis or cspace file and method != ee then only
        # mooseAddChemSolver is called.
        ret = _moose.loadModelInternal(filename, modelpath, "ee")
        method = "ee"
        if sc in ["gssa", "gillespie", "stochastic", "gsolve"]:
            method = "gssa"
        elif sc in ["gsl", "deterministic", "rungekutta", "rk5", "rk"]:
            method = "gsl"

        if method != "ee":
            _chemUtil.add_Delete_ChemicalSolver.mooseAddChemSolver(modelpath, method)
        return ret
    raise NameError("Unknown model extenstion '%s'" % ext)

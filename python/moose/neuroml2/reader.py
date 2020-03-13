# -*- coding: utf-8 -*-

# Description: NeuroML2 reader.
#     Implementation of reader for NeuroML 2 models.
#     TODO: handle morphologies of more than one segment...
#
# Author: Subhasis Ray, Padraig Gleeson
# Maintainer: Dilawar Singh <dilawars@ncbs.res.in>
# Created: Wed Jul 24 15:55:54 2013 (+0530)
#
# Notes:
#    For update/log, please see git-blame documentation or browse the github
#    repo https://github.com/BhallaLab/moose-core

import math
import numpy as np
from moose.neuroml2.hhfit import exponential2
from moose.neuroml2.hhfit import sigmoid2
from moose.neuroml2.hhfit import linoid2
from moose.neuroml2.units import SI
import moose

import logging

logger_ = logging.getLogger("moose.nml2")
logger_.setLevel(logging.INFO)

try:
    import neuroml.loaders as loaders
    from pyneuroml import pynml
except Exception as e:
    print(e)
    print("********************************************************************")
    print("* ")
    print("*  Please install libNeuroML & pyNeuroML: ")
    print("*    pip install libneuroml")
    print("*    pip install pyNeuroML")
    print("* ")
    print("*  Requirement for this is lxml: ")
    print("*    apt-get install python-lxml")
    print("* ")
    print("********************************************************************")

# these are the gates available. These are prefixed by 'gate' in C++ codebase.
_validMooseHHGateIds = ["X", "Y", "Z"]


def _unique(ls):
    res = []
    for l in ls:
        if l not in res:
            res.append(l)
    return res


def _whichGate(chan):
    global _validMooseHHGateIds
    c = chan.name[-1]
    assert c in _validMooseHHGateIds
    return c

def _setAttrFromNMLAttr(mObj, mAttr, nObj, nAttr, convertToSI=False):
    """Set MOOSE's object attribute from NML attribute

    :param mObj: MOOSE object.
    :param mAttr: MOOSE object's attribute
    :param nObj: NML2 Object
    :param nAttr: NML2 Object attribute.
    :param convertToSI: If `True` convert value to si unit.
    """
    if not hasattr(nObj, nAttr):
        return 
    if not hasattr(mObj, mAttr):
        return
    val = SI(getattr(nObj, nAttr)) if convertToSI else getattr(nObj, nAttr)
    setattr(mObj, mAttr, val)

def _pairNmlGateWithMooseGates(mGates, nmlGates):
    """Return moose gate id from nml.HHGate
    """
    global _validMooseHHGateIds
    # deep copy
    mooseGatesMap = {_whichGate(x): x for x in mGates}
    availableMooseGates = _validMooseHHGateIds[:]
    mapping = {}
    for nmlGate in nmlGates:
        if nmlGate is None:
            continue
        if (
            hasattr(nmlGate, "id")
            and nmlGate.id
            and nmlGate.id.upper() in availableMooseGates
        ):
            mapping[nmlGate.id.upper()] = nmlGate
            availableMooseGates.remove(nmlGate.id.upper())
        else:
            mapping[availableMooseGates.pop(0)] = nmlGate

    # Now replace 'X', 'Y', 'Z' with moose gates.
    return [(mooseGatesMap[x], mapping[x]) for x in mapping]


def _isConcDep(ct):
    """_isConcDep
    Check if componet is dependant on concentration. Most HHGates are
    dependant on voltage.

    :param ct: ComponentType
    :type ct: nml.ComponentType 

    :return: True if Component is depenant on conc, False otherwise.
    """
    if not hasattr(ct, "extends"):
        return False
    if "ConcDep" in ct.extends:
        return True
    return False


def _findCaConc():
    """_findCaConc
    Find a suitable CaConc for computing HHGate tables.
    This is a hack, though it is likely to work in most cases. 
    """
    caConcs = moose.wildcardFind("/library/##[TYPE=CaConc]")
    assert len(caConcs) == 1, "No moose.CaConc found." + \
        " Currently moose supports HHChannel which depends only " + \
        " on moose.CaConc. %s" % str(caConcs)
    return caConcs[0]


def sarea(comp):
    """
    Return the surface area of compartment from length and
    diameter.

    Parameters
    ----------
    comp : Compartment instance.

    Returns
    -------
    s : float
        surface area of `comp`.

    """
    if comp.length > 0:
        return comp.length * comp.diameter * np.pi
    else:
        return comp.diameter * comp.diameter * np.pi


def xarea(comp):
    """
    Return the cross sectional area from diameter of the
    compartment. How to do it for spherical compartment?"""
    return comp.diameter * comp.diameter * np.pi / 4.0


def setRa(comp, resistivity):
    """Calculate total raxial from specific value `resistivity`"""
    if comp.length > 0:
        comp.Ra = resistivity * comp.length / xarea(comp)
    else:
        comp.Ra = resistivity * 8.0 / (comp.diameter * np.pi)


def setRm(comp, condDensity):
    """Set membrane resistance"""
    comp.Rm = 1 / (condDensity * sarea(comp))


def setEk(comp, erev):
    """Set reversal potential"""
    comp.setEm(erev)


def getSegments(nmlcell, component, sg_to_segments):
    """Get the list of segments the `component` is applied to"""
    sg = component.segment_groups
    # seg = component.segment
    if sg is None:
        segments = nmlcell.morphology.segments
    elif sg == "all":  # Special case
        segments = [seg for seglist in sg_to_segments.values() for seg in seglist]
    else:
        segments = sg_to_segments[sg]

    unique_segs = []
    unique_segs_ids = []
    for s in segments:
        if not s.id in unique_segs_ids:
            unique_segs.append(s)
            unique_segs_ids.append(s.id)
    return unique_segs


class NML2Reader(object):
    """Reads NeuroML2 and creates MOOSE model. 

    NML2Reader.read(filename) reads an NML2 model under `/library`
    with the toplevel name defined in the NML2 file.

    Example:

    >>> from moose import neuroml2 as nml
    >>> reader = nml.NML2Reader()
    >>> reader.read('moose/neuroml2/test_files/Purk2M9s.nml')

    creates a passive neuronal morphology `/library/Purk2M9s`.
    """

    def __init__(self, verbose=False):
        # micron is the default length unit
        self.lunit = 1e-6

        self.verbose = verbose
        self.doc = None
        self.filename = None

        self.nml_cells_to_moose = {}  # NeuroML object to MOOSE object
        self.nml_segs_to_moose = {}  # NeuroML object to MOOSE object
        self.nml_chans_to_moose = {}  # NeuroML object to MOOSE object
        self.nml_concs_to_moose = {}  # NeuroML object to MOOSE object
        self.moose_to_nml = {}  # Moose object to NeuroML object
        self.proto_cells = {}  # map id to prototype cell in moose
        self.proto_chans = {}  # map id to prototype channels in moose
        self.proto_pools = {}  # map id to prototype pools (Ca2+, Mg2+)
        self.includes = {}  # Included files mapped to other readers

        # /library may have alreay been created.
        if moose.exists("/library"):
            self.lib = moose.element("/library")
        else:
            self.lib = moose.Neutral("/library")

        self.id_to_ionChannel = {}

        # nml cell to dict - the dict maps segment groups to segments
        self._cell_to_sg = {}

        self.cells_in_populations = {}
        self.pop_to_cell_type = {}
        self.seg_id_to_comp_name = {}
        self.paths_to_chan_elements = {}

        # Just in case.
        self._variables = {}

    def read(self, filename, symmetric=True):
        self.doc = loaders.read_neuroml2_file(
            filename, include_includes=True, verbose=self.verbose
        )

        if self.verbose:
            print("Parsed NeuroML2 file: %s" % filename)
        self.filename = filename

        if len(self.doc.networks) >= 1:
            self.network = self.doc.networks[0]

            moose.celsius = self._getTemperature()

        self.importConcentrationModels(self.doc)
        self.importIonChannels(self.doc)
        self.importInputs(self.doc)

        for cell in self.doc.cells:
            self.createCellPrototype(cell, symmetric=symmetric)

        for iaf in self.doc.iaf_cells:
            self.createIAFCellPrototype(iaf)

        if len(self.doc.networks) >= 1:
            self.createPopulations()
            self.createInputs()
        print("Read all from %s" % filename)

    def _getTemperature(self):
        if self.network.type == "networkWithTemperature":
            return SI(self.network.temperature)
        else:
            return 0  # Why not, if there's no temp dependence in nml..?

    def getCellInPopulation(self, pop_id, index):
        return self.cells_in_populations[pop_id][index]

    def getComp(self, pop_id, cellIndex, segId):
        if pop_id not in self.pop_to_cell_type:
            logger_.error("%s is not in populations: %s" % (pop_id 
                , str(list(self.pop_to_cell_type.keys()))))
            raise LookupError("%s not found" % pop_id)

        cellType = self.pop_to_cell_type[pop_id]
        if cellType not in self.seg_id_to_comp_name:
            logger_.error("%s not found in %s.compartments: %s" % (cellType
                , pop_id, str(list(self.seg_id_to_comp_name.keys()))))
            raise LookupError("%s not found" % cellType)

        compt = self.seg_id_to_comp_name[cellType]
        if segId not in compt:
            logger_.error("%s not found in %s.%s.segments: %s" % (compt
                , pop_id, cellType, str(list(compt.keys()))))
            raise LookupError("%s not found" % segId)
        return moose.element(
            "%s/%s/%s/%s"
            % (
                self.lib.path,
                pop_id,
                cellIndex,
                self.seg_id_to_comp_name[self.pop_to_cell_type[pop_id]][segId],
            )
        )

    def createPopulations(self):
        for pop in self.network.populations:
            # Sometime NML2 returns None instead of 0
            logger_.info("Adding population %s" % pop)
            pop.size = 0 if pop.size is None else pop.size
            mpop = moose.Neutral("%s/%s" % (self.lib.path, pop.id))
            self.cells_in_populations[pop.id] = {}

            # Either population have size of instances
            for i in range(pop.size):
                self.pop_to_cell_type[pop.id] = pop.component
                chid = moose.copy(self.proto_cells[pop.component], mpop, "%d"%i)
                self.cells_in_populations[pop.id][i] = chid
                logger_.info("Created %s instances of %s (Type %s)"
                    % (chid, pop.id, pop.component)
                )

            # Add instance of population.
            for i, instance in enumerate(pop.instances):
                self.pop_to_cell_type[pop.id] = pop.component
                chid = moose.copy(self.proto_cells[pop.component], mpop, '%d'%instance.id)
                self.cells_in_populations[pop.id][instance.id] = chid
                logger_.info("Created %s instances of %s (Type %s)"
                    % (chid, pop.id, pop.component)
                )

    def getInput(self, input_id):
        return moose.element("%s/inputs/%s" % (self.lib.path, input_id))

    def createInputs(self):
        for el in self.network.explicit_inputs:
            pop_id = el.target.split("[")[0]
            i = el.target.split("[")[1].split("]")[0]
            seg_id = 0
            if "/" in el.target:
                seg_id = el.target.split("/")[1]
            input = self.getInput(el.input)
            moose.connect(input, "output", self.getComp(pop_id, i, seg_id), "injectMsg")

        for il in self.network.input_lists:
            for ii in il.input:
                input = self.getInput(il.component)
                moose.connect(
                    input,
                    "output",
                    self.getComp(
                        il.populations, ii.get_target_cell_id(), ii.get_segment_id()
                    ),
                    "injectMsg",
                )

    def createIAFCellPrototype(self, iaf):
        """FIXME: Not tested.
        """
        mLIF = moose.LIF("%s/%s" % (self.lib.path, iaf.id))
        _setAttrFromNMLAttr(mLIF, 'vReset', iaf, 'reset', True)
        _setAttrFromNMLAttr(mLIF, 'thres', iaf, 'thres', True)
        _setAttrFromNMLAttr(mLIF, 'refractoryPeriod', iaf, 'refrac', True)
        _setAttrFromNMLAttr(mLIF, 'Cm', iaf, 'C', True)
        _setAttrFromNMLAttr(mLIF, 'Ra', iaf, 'Ra', True)

        if iaf.leak_conductance:
            mLIF.Rm = 1.0/SI(iaf.leak_conductance)

        if hasattr(iaf, 'leak_reversal'):
            logger_.warning("moose.LIF does not supprot 'leakReversal' ")

        self.proto_cells[iaf.id] = mLIF
        self.nml_cells_to_moose[iaf.id] = mLIF
        self.moose_to_nml[mLIF] = iaf

        # IAF cells have no morphology. Only one element. We need to create an element which
        # can recieve input.
        self.seg_id_to_comp_name[iaf.id] = {0:''}

        return iaf, mLIF

    def createCellPrototype(self, cell, symmetric=True):
        """To be completed - create the morphology, channels in prototype"""
        nrn = moose.Neuron("%s/%s" % (self.lib.path, cell.id))
        self.proto_cells[cell.id] = nrn
        self.nml_cells_to_moose[cell.id] = nrn
        self.moose_to_nml[nrn] = cell
        self.createMorphology(cell, nrn, symmetric=symmetric)
        self.importBiophysics(cell, nrn)
        return cell, nrn

    def createMorphology(self, nmlcell, moosecell, symmetric=True):
        """Create the MOOSE compartmental morphology in `moosecell` using the
        segments in NeuroML2 cell `nmlcell`. Create symmetric
        compartments if `symmetric` is True.

        """
        morphology = nmlcell.morphology
        if morphology is None:
            logger_.warning("%s has no morphology?" % nmlcell)
            return

        segments = morphology.segments

        id_to_segment = dict([(seg.id, seg) for seg in segments])
        if symmetric:
            compclass = moose.SymCompartment
        else:
            compclass = moose.Compartment
        # segment names are used as compartment names - assuming
        # naming convention does not clash with that in MOOSE
        cellpath = moosecell.path
        id_to_comp = {}
        self.seg_id_to_comp_name[nmlcell.id] = {}
        for seg in segments:
            if seg.name is not None:
                id_to_comp[seg.id] = compclass("%s/%s" % (cellpath, seg.name))
                self.seg_id_to_comp_name[nmlcell.id][seg.id] = seg.name
            else:
                name = "comp_%s" % seg.id
                id_to_comp[seg.id] = compclass("%s/%s" % (cellpath, name))
                self.seg_id_to_comp_name[nmlcell.id][seg.id] = name
        # Now assign the positions and connect up axial resistance
        if not symmetric:
            src, dst = "axial", "raxial"
        else:
            src, dst = "proximal", "distal"
        for segid, comp in id_to_comp.items():
            segment = id_to_segment[segid]
            try:
                parent = id_to_segment[segment.parent.segments]
            except AttributeError:
                parent = None
            self.moose_to_nml[comp] = segment
            self.nml_segs_to_moose[segment.id] = comp
            p0 = segment.proximal
            if p0 is None:
                if parent:
                    p0 = parent.distal
                else:
                    raise Exception(
                        "No proximal point and no parent segment for segment: name=%s, id=%s"
                        % (segment.name, segment.id)
                    )
            comp.x0, comp.y0, comp.z0 = (
                x * self.lunit for x in map(float, (p0.x, p0.y, p0.z))
            )
            p1 = segment.distal
            comp.x, comp.y, comp.z = (
                x * self.lunit for x in map(float, (p1.x, p1.y, p1.z))
            )
            comp.length = np.sqrt(
                (comp.x - comp.x0) ** 2
                + (comp.y - comp.y0) ** 2
                + (comp.z - comp.z0) ** 2
            )
            # This can pose problem with moose where both ends of
            # compartment have same diameter. We are averaging the two
            # - may be splitting the compartment into two is better?
            comp.diameter = (float(p0.diameter) + float(p1.diameter)) * self.lunit / 2
            if parent:
                pcomp = id_to_comp[parent.id]
                moose.connect(comp, src, pcomp, dst)
        sg_to_segments = {}
        for sg in morphology.segment_groups:
            sg_to_segments[sg.id] = [id_to_segment[m.segments] for m in sg.members]
        for sg in morphology.segment_groups:
            if not sg.id in sg_to_segments:
                sg_to_segments[sg.id] = []
            for inc in sg.includes:
                for cseg in sg_to_segments[inc.segment_groups]:
                    sg_to_segments[sg.id].append(cseg)

        if not "all" in sg_to_segments:
            sg_to_segments["all"] = [s for s in segments]

        self._cell_to_sg[nmlcell.id] = sg_to_segments
        return id_to_comp, id_to_segment, sg_to_segments

    def importBiophysics(self, nmlcell, moosecell):
        """Create the biophysical components in moose Neuron `moosecell`
        according to NeuroML2 cell `nmlcell`."""
        bp = nmlcell.biophysical_properties
        if bp is None:
            print(
                "Warning: %s in %s has no biophysical properties"
                % (nmlcell.id, self.filename)
            )
            return
        self.importMembraneProperties(nmlcell, moosecell, bp.membrane_properties)
        self.importIntracellularProperties(
            nmlcell, moosecell, bp.intracellular_properties
        )

    def importMembraneProperties(self, nmlcell, moosecell, mp):
        """Create the membrane properties from nmlcell in moosecell"""
        if self.verbose:
            print("Importing membrane properties")
        self.importCapacitances(nmlcell, moosecell, mp.specific_capacitances)
        self.importChannelsToCell(nmlcell, moosecell, mp)
        self.importInitMembPotential(nmlcell, moosecell, mp)

    def importCapacitances(self, nmlcell, moosecell, specificCapacitances):
        sg_to_segments = self._cell_to_sg[nmlcell.id]
        for specific_cm in specificCapacitances:
            cm = SI(specific_cm.value)
            for seg in sg_to_segments[specific_cm.segment_groups]:
                comp = self.nml_segs_to_moose[seg.id]
                comp.Cm = sarea(comp) * cm

    def importInitMembPotential(self, nmlcell, moosecell, membraneProperties):
        sg_to_segments = self._cell_to_sg[nmlcell.id]
        for imp in membraneProperties.init_memb_potentials:
            initv = SI(imp.value)
            for seg in sg_to_segments[imp.segment_groups]:
                comp = self.nml_segs_to_moose[seg.id]
                comp.initVm = initv

    def importIntracellularProperties(self, nmlcell, moosecell, properties):
        self.importAxialResistance(nmlcell, properties)
        self.importSpecies(nmlcell, properties)

    def importSpecies(self, nmlcell, properties):
        sg_to_segments = self._cell_to_sg[nmlcell.id]
        for species in properties.species:
            if (species.concentration_model is not None) and (
                species.concentration_model not in self.proto_pools
            ):
                continue
            segments = getSegments(nmlcell, species, sg_to_segments)
            for seg in segments:
                comp = self.nml_segs_to_moose[seg.id]
                self.copySpecies(species, comp)

    def copySpecies(self, species, compartment):
        """Copy the prototype pool `species` to compartment. Currently only
        decaying pool of Ca2+ supported"""
        proto_pool = None
        if species.concentration_model in self.proto_pools:
            proto_pool = self.proto_pools[species.concentration_model]
        else:
            for innerReader in self.includes.values():
                if species.concentration_model in innerReader.proto_pools:
                    proto_pool = innerReader.proto_pools[species.concentration_model]
                    break
        if not proto_pool:
            raise Exception(
                "No prototype pool for %s referred to by %s"
                % (species.concentration_model, species.id)
            )
        pool_id = moose.copy(proto_pool, compartment, species.id)
        pool = moose.element(pool_id)
        pool.B = pool.B / (
            np.pi
            * compartment.length
            * (0.5 * compartment.diameter + pool.thick)
            * (0.5 * compartment.diameter - pool.thick)
        )
        return pool

    def importAxialResistance(self, nmlcell, intracellularProperties):
        sg_to_segments = self._cell_to_sg[nmlcell.id]
        for r in intracellularProperties.resistivities:
            segments = getSegments(nmlcell, r, sg_to_segments)
            for seg in segments:
                comp = self.nml_segs_to_moose[seg.id]
                setRa(comp, SI(r.value))

    def isPassiveChan(self, chan):
        if chan.type == "ionChannelPassive":
            return True
        if hasattr(chan, "gates"):
            return len(chan.gate_hh_rates) + len(chan.gates) == 0
        return False

    rate_fn_map = {
        "HHExpRate": exponential2,
        "HHSigmoidRate": sigmoid2,
        "HHSigmoidVariable": sigmoid2,
        "HHExpLinearRate": linoid2,
    }

    def calculateRateFn(self, ratefn, mgate, vmin, vmax, tablen=3000, vShift="0mV"):
        """Returns A / B table from ngate."""

        tab = np.linspace(vmin, vmax, tablen)
        if self._is_standard_nml_rate(ratefn):
            midpoint, rate, scale = map(
                SI, (ratefn.midpoint, ratefn.rate, ratefn.scale)
            )
            return self.rate_fn_map[ratefn.type](tab, rate, scale, midpoint)

        for ct in self.doc.ComponentType:
            if ratefn.type != ct.name:
                continue
            logger_.info("Using %s to evaluate rate" % ct.name)
            if not _isConcDep(ct):
                return self._computeRateFn(ct, tab)
            else:
                ca = _findCaConc()
                if _whichGate(mgate) != "Z":
                    raise RuntimeWarning(
                        "Concentration dependant gate "
                        " should use gateZ of moose.HHChannel. "
                        " If you know what you are doing, ignore this "
                        " warning. "
                    )
                return self._computeRateFnCa(ca, ct, tab, vShift=vShift)

    def _computeRateFnCa(self, ca, ct, tab, vShift):
        rate = []
        for v in tab:
            req_vars = {
                ca.name: "%sV" % v,
                "vShift": vShift,
                "temperature": self._getTemperature(),
            }
            req_vars.update(self._variables)
            vals = pynml.evaluate_component(ct, req_variables=req_vars)
            """print(vals)"""
            if "x" in vals:
                rate.append(vals["x"])
            if "t" in vals:
                rate.append(vals["t"])
            if "r" in vals:
                rate.append(vals["r"])
        return np.array(rate)

    def _computeRateFn(self, ct, tab, vShift=0):
        rate = []
        for v in tab:
            req_vars = {
                "v": "%sV" % v,
                "vShift": vShift,
                "temperature": self._getTemperature(),
            }
            req_vars.update(self._variables)
            vals = pynml.evaluate_component(ct, req_variables=req_vars)
            """print(vals)"""
            if "x" in vals:
                rate.append(vals["x"])
            if "t" in vals:
                rate.append(vals["t"])
            if "r" in vals:
                rate.append(vals["r"])
        return np.array(rate)

    def importChannelsToCell(self, nmlcell, moosecell, membrane_properties):
        sg_to_segments = self._cell_to_sg[nmlcell.id]
        for chdens in (
            membrane_properties.channel_densities
            + membrane_properties.channel_density_v_shifts
        ):
            segments = getSegments(nmlcell, chdens, sg_to_segments)
            condDensity = SI(chdens.cond_density)
            erev = SI(chdens.erev)
            try:
                ionChannel = self.id_to_ionChannel[chdens.ion_channel]
            except KeyError:
                print("No channel with id", chdens.ion_channel)
                continue

            if self.verbose:
                print(
                    "Setting density of channel %s in %s to %s; erev=%s (passive: %s)"
                    % (
                        chdens.id,
                        segments,
                        condDensity,
                        erev,
                        self.isPassiveChan(ionChannel),
                    )
                )

            if self.isPassiveChan(ionChannel):
                for seg in segments:
                    comp = self.nml_segs_to_moose[seg.id]
                    setRm(comp, condDensity)
                    setEk(comp, erev)
            else:
                for seg in segments:
                    self.copyChannel(
                        chdens, self.nml_segs_to_moose[seg.id], condDensity, erev
                    )
            """moose.le(self.nml_segs_to_moose[seg.id])
            moose.showfield(self.nml_segs_to_moose[seg.id], field="*", showtype=True)"""

    def copyChannel(self, chdens, comp, condDensity, erev):
        """Copy moose prototype for `chdens` condutcance density to `comp`
        compartment.

        """
        proto_chan = None
        if chdens.ion_channel in self.proto_chans:
            proto_chan = self.proto_chans[chdens.ion_channel]
        else:
            for innerReader in self.includes.values():
                if chdens.ionChannel in innerReader.proto_chans:
                    proto_chan = innerReader.proto_chans[chdens.ion_channel]
                    break
        if not proto_chan:
            raise Exception(
                "No prototype channel for %s referred to by %s"
                % (chdens.ion_channel, chdens.id)
            )

        if self.verbose:
            print(
                "Copying %s to %s, %s; erev=%s" % (chdens.id, comp, condDensity, erev)
            )
        orig = chdens.id
        chid = moose.copy(proto_chan, comp, chdens.id)
        chan = moose.element(chid)
        els = list(self.paths_to_chan_elements.keys())
        for p in els:
            pp = p.replace("%s/" % chdens.ion_channel, "%s/" % orig)
            self.paths_to_chan_elements[pp] = self.paths_to_chan_elements[p].replace(
                "%s/" % chdens.ion_channel, "%s/" % orig
            )
        # print(self.paths_to_chan_elements)
        chan.Gbar = sarea(comp) * condDensity
        chan.Ek = erev
        moose.connect(chan, "channel", comp, "channel")
        return chan

    """
    def importIncludes(self, doc):        
        for include in doc.include:
            if self.verbose:
                print(self.filename, 'Loading include', include)
            error = None
            inner = NML2Reader(self.verbose)
            paths = [include.href, os.path.join(os.path.dirname(self.filename), include.href)]
            for path in paths:
                try:
                    inner.read(path)                    
                    if self.verbose:
                        print(self.filename, 'Loaded', path, '... OK')
                except IOError as e:
                    error = e
                else:
                    self.includes[include.href] = inner
                    self.id_to_ionChannel.update(inner.id_to_ionChannel)
                    self.nml_to_moose.update(inner.nml_to_moose)
                    self.moose_to_nml.update(inner.moose_to_nml)
                    error = None
                    break
            if error:
                print(self.filename, 'Last exception:', error)
                raise IOError('Could not read any of the locations: %s' % (paths))"""

    def _is_standard_nml_rate(self, rate):
        return (
            rate.type == "HHExpLinearRate"
            or rate.type == "HHExpRate"
            or rate.type == "HHSigmoidRate"
            or rate.type == "HHSigmoidVariable"
        )

    def createHHChannel(self, chan, vmin=-150e-3, vmax=100e-3, vdivs=5000):
        path = "%s/%s" % (self.lib.path, chan.id)
        if moose.exists(path):
            mchan = moose.element(path)
        else:
            mchan = moose.HHChannel(path)
        mgates = [moose.element(g) for g in [mchan.gateX, mchan.gateY, mchan.gateZ]]

        # We handle only up to 3 gates in HHCHannel
        assert len(chan.gate_hh_rates) <= 3, "No more than 3 gates"

        if self.verbose:
            print(
                "== Creating channel: %s (%s) -> %s (%s)"
                % (chan.id, chan.gate_hh_rates, mchan, mgates)
            )

        all_gates = chan.gates + chan.gate_hh_rates

        # If user set bnml channels' id to 'x', 'y' or 'z' then pair this gate
        # with moose.HHChannel's gateX, gateY, gateZ respectively. Else pair
        # them with gateX, gateY, gateZ acording to list order.
        for mgate, ngate in _pairNmlGateWithMooseGates(mgates, all_gates):
            self._addGateToHHChannel(chan, mchan, mgate, ngate, vmin, vmax, vdivs)
        logger_.debug("== Created %s for %s" % (mchan.path, chan.id))
        return mchan

    def _addGateToHHChannel(self, chan, mchan, mgate, ngate, vmin, vmax, vdivs):
        """Add gateX, gateY, gateZ etc to moose.HHChannel (mchan). 

        Each gate can be voltage dependant and/or concentration dependant.
        Only caConc dependant channels are supported.
        """

        # set mgate.Xpower, .Ypower etc.
        setattr(mchan, _whichGate(mgate) + "power", ngate.instances)

        mgate.min = vmin
        mgate.max = vmax
        mgate.divs = vdivs

        # Note by Padraig:
        # ---------------
        # I saw only examples of GateHHRates in HH-channels, the meaning of
        # forwardRate and reverseRate and steadyState are not clear in the
        # classes GateHHRatesInf, GateHHRatesTau and in FateHHTauInf the
        # meaning of timeCourse and steady state is not obvious. Is the last
        # one # refering to tau_inf and m_inf??
        fwd = ngate.forward_rate
        rev = ngate.reverse_rate

        self.paths_to_chan_elements["%s/%s" % (chan.id, ngate.id)] = "%s/%s" % (
            chan.id,
            mgate.name,
        )

        q10_scale = 1
        if ngate.q10_settings:
            if ngate.q10_settings.type == "q10Fixed":
                q10_scale = float(ngate.q10_settings.fixed_q10)
            elif ngate.q10_settings.type == "q10ExpTemp":
                q10_scale = math.pow(
                    float(ngate.q10_settings.q10_factor),
                    (self._getTemperature() - SI(ngate.q10_settings.experimental_temp))
                    / 10,
                )
                logger_.debug(
                    "Q10: %s; %s; %s; %s"
                    % (
                        ngate.q10_settings.q10_factor,
                        self._getTemperature(),
                        SI(ngate.q10_settings.experimental_temp),
                        q10_scale,
                    )
                )
            else:
                raise Exception(
                    "Unknown Q10 scaling type %s: %s"
                    % (ngate.q10_settings.type, ngate.q10_settings)
                )
        logger_.info(
            " === Gate: %s; %s; %s; %s; %s; scale=%s"
            % (ngate.id, mgate.path, mchan.Xpower, fwd, rev, q10_scale)
        )

        if (fwd is not None) and (rev is not None):
            # Note: MOOSE HHGate are either voltage of concentration
            # dependant. Here we figure out if nml description of gate is
            # concentration dependant or not.
            alpha = self.calculateRateFn(fwd, mgate, vmin, vmax, vdivs)
            beta = self.calculateRateFn(rev, mgate, vmin, vmax, vdivs)

            mgate.tableA = q10_scale * (alpha)
            mgate.tableB = q10_scale * (alpha + beta)

        # Assuming the meaning of the elements in GateHHTauInf ...
        if (
            hasattr(ngate, "time_course")
            and hasattr(ngate, "steady_state")
            and (ngate.time_course is not None)
            and (ngate.steady_state is not None)
        ):
            tau = ngate.time_course
            inf = ngate.steady_state
            tau = self.calculateRateFn(tau, mgate, vmin, vmax, vdivs)
            inf = self.calculateRateFn(inf, mgate, vmin, vmax, vdivs)
            mgate.tableA = q10_scale * (inf / tau)
            mgate.tableB = q10_scale * (1 / tau)

        if (
            hasattr(ngate, "steady_state")
            and (ngate.time_course is None)
            and (ngate.steady_state is not None)
        ):
            inf = ngate.steady_state
            tau = 1 / (alpha + beta)
            if inf is not None:
                inf = self.calculateRateFn(inf, vmin, vmax, vdivs)
                mgate.tableA = q10_scale * (inf / tau)
                mgate.tableB = q10_scale * (1 / tau)

    def createPassiveChannel(self, chan):
        mchan = moose.Leakage("%s/%s" % (self.lib.path, chan.id))
        if self.verbose:
            print(self.filename, "Created", mchan.path, "for", chan.id)
        return mchan

    def importInputs(self, doc):
        minputs = moose.Neutral("%s/inputs" % (self.lib.path))

        for pg_nml in doc.pulse_generators:
            assert pg_nml.id
            pg = moose.PulseGen("%s/%s" % (minputs.path, pg_nml.id))
            pg.firstDelay = SI(pg_nml.delay)
            pg.firstWidth = SI(pg_nml.duration)
            pg.firstLevel = SI(pg_nml.amplitude)
            pg.secondDelay = 1e9

    def importIonChannels(self, doc, vmin=-150e-3, vmax=100e-3, vdivs=5000):
        if self.verbose:
            print(self.filename, "Importing the ion channels")

        for chan in doc.ion_channel + doc.ion_channel_hhs:
            if chan.type == "ionChannelHH":
                mchan = self.createHHChannel(chan)
            elif self.isPassiveChan(chan):
                mchan = self.createPassiveChannel(chan)
            else:
                mchan = self.createHHChannel(chan)

            assert chan.id, "Empty id is not allowed"
            self.id_to_ionChannel[chan.id] = chan
            self.nml_chans_to_moose[chan.id] = mchan
            self.proto_chans[chan.id] = mchan
            if self.verbose:
                print(
                    self.filename,
                    "Created ion channel",
                    mchan.path,
                    "for",
                    chan.type,
                    chan.id,
                )

    def importConcentrationModels(self, doc):
        for concModel in doc.decaying_pool_concentration_models:
            #  proto = self.createDecayingPoolConcentrationModel(concModel)
            self.createDecayingPoolConcentrationModel(concModel)

    def createDecayingPoolConcentrationModel(self, concModel):
        """Create prototype for concentration model"""
        assert concModel.id, "Empty id is not allowed"
        name = concModel.id
        if hasattr(concModel, "name") and concModel.name is not None:
            name = concModel.name
        ca = moose.CaConc("%s/%s" % (self.lib.path, name))

        ca.CaBasal = SI(concModel.resting_conc)
        ca.tau = SI(concModel.decay_constant)
        ca.thick = SI(concModel.shell_thickness)

        # B = 5.2e-6/(Ad) where A is the area of the shell and d is thickness - must divide by shell volume when copying
        ca.B = 5.2e-6
        self.proto_pools[concModel.id] = ca
        self.nml_concs_to_moose[concModel.id] = ca
        self.moose_to_nml[ca] = concModel
        logger_.debug(
            "Created moose element: %s for nml conc %s" % (ca.path, concModel.id)
        )

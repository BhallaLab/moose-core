# -*- coding: utf-8 -*-

from __future__ import print_function, division, absolute_import

# Description: NeuroML2 reader.
#     Implementation of reader for NeuroML 2 models.
#     TODO: handle morphologies of more than one segment...
# Author: Subhasis Ray, Padraig Gleeson
# Maintainer: Padraig Gleeson, Subhasis Ray
# Created: Wed Jul 24 15:55:54 2013 (+0530)
# Notes:
#    For update/log, please see git-blame documentation or browse the github
#    repo https://github.com/BhallaLab/moose-core

import os
import math
import logging
import numpy as np
from collections import defaultdict

from moose.neuroml2.hhfit import exponential2
from moose.neuroml2.hhfit import sigmoid2
from moose.neuroml2.hhfit import linoid2
from moose.neuroml2.units import SI


import moose


logger_ = logging.getLogger("moose.nml2")
logger_.setLevel(logging.DEBUG)

nml_not_available_msg = ""

try:
    import neuroml as nml
    import pyneuroml.pynml as pynml
except ImportError as error:
    raise ImportError(
        "Could not import neuroml/pyneuroml. "
        "Please make sure you have pyneuroml installed "
        "(`pip install pyneuroml`)"
    ) from error


from moose.neuroml2.units import SI


PREDEFINED_RATEFN_MAP = {
    "HHExpRate": exponential2,
    "HHSigmoidRate": sigmoid2,
    "HHSigmoidVariable": sigmoid2,
    "HHExpLinearRate": linoid2,
}


def _write_flattened_nml(doc, outfile):
    """_write_flattened_nml
    Concat all NML2 read by moose and generate one flattened NML file.
    Only useful when debugging.

    :param doc: NML document (nml.doc)
    :param outfile: Name of the output file.
    """
    import neuroml.writers

    neuroml.writers.NeuroMLWriter.write(doc, outfile)
    logger_.debug("Wrote flattened NML model to %s" % outfile)


def _gates_sorted(all_gates):
    """_gates_sorted

    Parameters
    ----------
    all_gates (list)
        List of all moose.HHChannel.gates

    Notes
    -----
    If the id of gates are subset of 'x', 'y' or 'z' then sort them so they load in
    X, Y or Z gate respectively. Otherwise do not touch them i.e. first gate
    will be loaded into X, second into Y and so on.
    """
    allMooseGates = ["x", "y", "z"]
    allGatesDict = {g.id: g for g in all_gates}
    gateNames = [g.id.lower() for g in all_gates]
    if set(gateNames).issubset(set(allMooseGates)):
        sortedGates = []
        for gid in allMooseGates:
            sortedGates.append(allGatesDict.get(gid))
        return sortedGates
    return all_gates


def _unique(ls):
    res = []
    for l in ls:
        if l not in res:
            res.append(l)
    return res


def _isConcDep(ct):
    """_isConcDep
    Check if componet is dependent on concentration. Most HHGates are
    dependent on voltage.

    :param ct: ComponentType
    :type ct: nml.ComponentType

    :return: True if Component is depenant on conc, False otherwise.
    """
    # logger_.debug(f"{'#' * 10} EXTENDS {ct.extends}")
    if "ConcDep" in ct.extends:
        return True
    return False


def _findCaConcVariableName():
    """_findCaConcVariableName
    Find a suitable CaConc for computing HHGate tables.
    This is a hack, though it is likely to work in most cases.
    """
    caConcs = moose.wildcardFind("/library/##[TYPE=CaConc]")
    assert (
        len(caConcs) >= 1
    ), "No moose.CaConc found. Currently moose \
            supports HHChannel which depends only on moose.CaConc ."
    return caConcs[0].name


def sarea(comp):
    """Return the surface area (2 * pi * r * L) of compartment from
    length and diameter.

    :param comp: Compartment instance.
    :type comp: str
    :return: surface area of `comp`.
    :rtype: float

    """
    if comp.length > 0:
        return math.pi * comp.diameter * comp.length
    else:
        return math.pi * comp.diameter * comp.diameter


def xarea(compt):
    """xarea
    Return the cross sectional area (pi * r^2) from the diameter of the compartment.

    Note:
    ----
    How to do it for spherical compartment?

    :param compt: Compartment in moose.
    :type compt: moose.Compartment
    :return: cross sectional area.
    :rtype: float
    """
    return math.pi * (compt.diameter / 2.0) ** 2.0


def setRa(comp, resistivity):
    """Calculate total raxial from specific value `resistivity`"""
    if comp.length > 0:
        comp.Ra = resistivity * comp.length / xarea(comp)
    else:
        comp.Ra = resistivity * 4.0 / (comp.diameter * np.pi)


def setRm(comp, condDensity):
    """Set membrane resistance"""
    comp.Rm = 1 / (condDensity * sarea(comp))


def setEk(comp, erev):
    """Set reversal potential"""
    comp.setEm(erev)


def getSegments(nmlcell, component, sg_to_segments):
    """Get the list of segments the `component` is applied to"""
    sg = component.segment_groups
    if sg is None:
        segments = nmlcell.morphology.segments
    elif sg == "all":
        segments = [
            seg for seglist in sg_to_segments.values() for seg in seglist
        ]
    else:
        segments = sg_to_segments[sg]
    return _unique(segments)


class NML2Reader(object):
    """Reads NeuroML2 and creates MOOSE model.

    NML2Reader.read(filename) reads an NML2 model under `/library`
    with the toplevel name defined in the NML2 file.

    Example:

    >>> import moose
    >>> reader = moose.NML2Reader()
    >>> reader.read('moose/neuroml2/test_files/Purk2M9s.nml')

    creates a passive neuronal morphology `/library/Purk2M9s`.
    """

    def __init__(self, verbose=False):
        global logger_
        self.lunit = 1e-6  # micron is the default length unit
        self.verbose = verbose
        if self.verbose:
            logger_.setLevel(logging.DEBUG)
        self.doc = None
        self.filename = None
        self.nml_cells_to_moose = {}  # NeuroML object to MOOSE object
        self.nml_segs_to_moose = {}  # NeuroML object to MOOSE object
        self.nml_chans_to_moose = {}  # NeuroML object to MOOSE object
        self.nml_conc_to_moose = {}  # NeuroML object to MOOSE object
        self.moose_to_nml = {}  # Moose object to NeuroML object
        self.proto_cells = {}  # map id to prototype cell in moose
        self.proto_chans = {}  # map id to prototype channels in moose
        self.proto_pools = {}  # map id to prototype pools (Ca2+, Mg2+)
        self.includes = {}  # Included files mapped to other readers
        self.lib = moose.Neutral(
            "/library"
        )  # Keeps the prototypes: not simulated
        self.model = moose.Neutral("/model")  # Actual model: simulated
        self.id_to_ionChannel = {}
        self._cell_to_sg = (
            {}
        )  # nml cell to dict - the dict maps segment groups to segments
        self._variables = {}

        self.cells_in_populations = {}
        self.pop_to_cell_type = {}
        self.seg_id_to_comp_name = {}
        self.network = None

    def read(self, filename, symmetric=True):
        filename = os.path.realpath(filename)
        self.doc = nml.loaders.read_neuroml2_file(
            filename, include_includes=True, verbose=self.verbose
        )

        self.filename = filename

        logger_.info("Parsed the NeuroML2 file: %s" % filename)
        if self.verbose:
            _write_flattened_nml(self.doc, "%s__flattened.xml" % self.filename)

        if len(self.doc.networks) >= 1:
            self.network = self.doc.networks[0]
            moose.celsius = self._getTemperature()

        self.importConcentrationModels(self.doc)
        self.importIonChannels(self.doc)
        self.importInputs(self.doc)

        for cell in self.doc.cells:
            # logger_.debug(f"{'%' * 10} Creating cell prototype {cell}")
            self.createCellPrototype(cell, symmetric=symmetric)

        if len(self.doc.networks) >= 1:
            self.createPopulations()
            self.createInputs()

    def _getTemperature(self):
        if self.network is not None:
            if self.network.type == "networkWithTemperature":
                return SI(self.network.temperature)
            else:
                # Why not, if there's no temp dependence in nml..?
                return 0
        return SI("25")

    def getCellInPopulation(self, pop_id, index):
        return self.cells_in_populations[pop_id][index]

    def getComp(self, pop_id, cellIndex, segId):
        """Get moose compartment corresponding to the specified NeuroML element

        Parameters
        ----------
        pop_id : str
            Population ID in NeuroML
        cellIndex : str
            Index of cell in population
        segId : str
            Segment ID in NeuroML

        """
        comp_name = self.seg_id_to_comp_name[self.pop_to_cell_type[pop_id]][
            segId
        ]
        return moose.element(
            f"{self.model.path}/{self.network.id}/{pop_id}/{cellIndex}/{comp_name}"
        )

    def createPopulations(self):
        """Create populations of neurons in a network

        Create a container identified by `network.id` and create cell
        populations under that. The network container is created under
        `/model` as this is supposed to be an instantiation and not a
        prototype (the latter are created under `/library`).

        """
        net = moose.Neutral(f"{self.model.path}/{self.network.id}")
        for pop in self.network.populations:
            mpop = moose.Neutral(f"{net.path}/{pop.id}")
            self.pop_to_cell_type[pop.id] = pop.component
            logger_.info(
                f"Creating {pop.size} instances of cell "
                f"{pop.component} under {mpop.path}"
            )
            self.cells_in_populations[pop.id] = {
                ii: moose.copy(
                    self.proto_cells[pop.component],
                    mpop,
                    f"{ii}",
                )
                for ii in range(pop.size)
            }

    def getInput(self, input_id):
        """Returns input object (PulseGen) identified by `input_id`.

        Retrieves the object `/model/inputs/{input_id}`.

        Parameters
        ----------
        input_id : str
            NeuroML id of the object to be retrieved. Same as its name
            in MOOSE.

        """
        return moose.element(f"{self.model.path}/inputs/{input_id}")

    def createInputs(self):
        """Create inputs to model.

        Currently this assumes inputs are `PulseGen` objects and
        copies the prototypes for a network's `explicit_inputs` and
        `input_lists` from '/library/inputs' to '/model/inputs'. It
        also connects up the 'output' field of the `PulseGen object to
        the 'injectMsg' field of the target compartment.

        """
        inputs = moose.Neutral(f"{self.model.path}/inputs")
        for el in self.network.explicit_inputs:
            proto = moose.element(f"{self.lib.path}/inputs/{el.input}")
            pop_id = el.target.split("[")[0]
            ii = el.target.split("[")[1].split("]")[0]
            seg_id = 0
            if "/" in el.target:
                seg_id = el.target.split("/")[1]
            input_ = moose.copy(proto, inputs)
            moose.connect(
                input_, "output", self.getComp(pop_id, ii, seg_id), "injectMsg"
            )

        for il in self.network.input_lists:
            for ii in il.input:
                logger_.debug(f"il.component: {il.component}, input: {ii}")
                proto = moose.element(f"{self.lib.path}/inputs/{il.component}")
                input_ = moose.copy(proto, inputs)
                moose.connect(
                    input_,
                    "output",
                    self.getComp(
                        il.populations,
                        ii.get_target_cell_id(),
                        ii.get_segment_id(),
                    ),
                    "injectMsg",
                )

    def createCellPrototype(self, cell, symmetric=True):
        """To be completed - create the morphology, channels in prototype"""
        ep = f"{self.lib.path}/{cell.id}"
        nrn = moose.Neuron(ep)
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
        segments = morphology.segments
        id_to_segment = {seg.id: seg for seg in segments}
        if symmetric:
            compclass = moose.SymCompartment
            src, dst = "proximal", "distal"
        else:
            compclass = moose.Compartment
            src, dst = "axial", "raxial"
        # segment names are used as compartment names - assuming
        # naming convention does not clash with that in MOOSE
        cellpath = moosecell.path
        id_to_comp = {}
        self.seg_id_to_comp_name[nmlcell.id] = {}
        # create or access compartments by segment name (or id if name
        # is not set) and keep two-way mapping
        for seg in segments:
            name = seg.name if seg.name is not None else f"comp_{seg.id}"
            comp = compclass(f"{cellpath}/{name}")
            id_to_comp[seg.id] = comp
            self.seg_id_to_comp_name[nmlcell.id][seg.id] = name
        # Now assign the positions and connect up axial resistance
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
                        "No proximal point and no parent segment for segment: "
                        f"name={segment.name}, id={segment.id}"
                    )
            comp.x0, comp.y0, comp.z0 = (
                float(x) * self.lunit for x in (p0.x, p0.y, p0.z)
            )
            p1 = segment.distal
            comp.x, comp.y, comp.z = (
                float(x) * self.lunit for x in (p1.x, p1.y, p1.z)
            )
            comp.length = np.sqrt(
                (comp.x - comp.x0) ** 2
                + (comp.y - comp.y0) ** 2
                + (comp.z - comp.z0) ** 2
            )

            # NOTE: moose compartments are cylindrical (both ends of a
            # compartment have same diameter). So taking the average
            # of the two ends in case of truncated-cone.
            comp.diameter = (
                (float(p0.diameter) + float(p1.diameter)) * self.lunit / 2
            )
            if parent:
                pcomp = id_to_comp[parent.id]
                moose.connect(comp, src, pcomp, dst)
        # map segment-group to segments
        sg_to_segments = defaultdict(list)
        for sg in morphology.segment_groups:
            for m in sg.members:
                sg_to_segments[sg.id].append(id_to_segment[m.segments])
            for inc in sg.includes:
                for cseg in sg_to_segments[inc.segment_groups]:
                    sg_to_segments[sg.id].append(cseg)
        if "all" not in sg_to_segments:
            sg_to_segments["all"] = [s for s in segments]

        self._cell_to_sg[nmlcell.id] = sg_to_segments
        return id_to_comp, id_to_segment, sg_to_segments

    def importBiophysics(self, nmlcell, moosecell):
        """Create the biophysical components in moose Neuron `moosecell`
        according to NeuroML2 cell `nmlcell`."""
        bp = nmlcell.biophysical_properties
        if bp is None:
            logger_.info(
                "Warning: %s in %s has no biophysical properties"
                % (nmlcell.id, self.filename)
            )
            return
        self.importMembraneProperties(
            nmlcell, moosecell, bp.membrane_properties
        )
        self.importIntracellularProperties(
            nmlcell, moosecell, bp.intracellular_properties
        )

    def importMembraneProperties(self, nmlcell, moosecell, mp):
        """Create the membrane properties from nmlcell in moosecell"""
        if self.verbose:
            logger_.info("Importing membrane properties")
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
            # Developer note: Not sure if species.concentration_model should be
            # a nml element of just plain string. I was getting plain text from
            # nml file here.
            concModel = species.concentration_model
            if (concModel is not None) and (concModel not in self.proto_pools):
                logger_.warn("No concentrationModel '%s' found." % concModel)
                continue
            segments = getSegments(nmlcell, species, sg_to_segments)
            for seg in segments:
                comp = self.nml_segs_to_moose[seg.id]
                self.copySpecies(species, comp)

    def copySpecies(self, species, compartment):
        """Copy the prototype pool `species` to compartment. Currently only
        decaying pool of Ca2+ supported"""
        proto_pool = None
        concModel = species.concentration_model
        if concModel in self.proto_pools:
            proto_pool = self.proto_pools[concModel]
        else:
            for innerReader in self.includes.values():
                if concModel in innerReader.proto_pools:
                    proto_pool = innerReader.proto_pools[concModel]
                    break
        if not proto_pool:
            msg = "No prototype pool for %s referred to by %s" % (
                concModel,
                species.id,
            )
            logger_.error(msg)
            raise RuntimeError(msg)
        pool_id = moose.copy(proto_pool, compartment, species.id)
        pool = moose.element(pool_id)
        if compartment.length <= 0:
            vol = (
                4
                * np.pi
                * (
                    0.5 * compartment.diameter**3
                    - (0.5 * compartment.diameter - pool.thick) ** 3
                )
                / 3
            )
        else:
            vol = (
                np.pi
                * compartment.length
                * (0.5 * compartment.diameter + pool.thick)
                * (0.5 * compartment.diameter - pool.thick)
            )
        pool.B = pool.B / vol
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

    def evaluate_moose_component(self, ct, variables):
        print("[INFO ] Not implemented.")
        return False

    def rateFnFromFormula(
        self, ratefn, ctype, vtab, vshift="0.0V", param_tabs=None
    ):
        """Compute rate values from formula provided in NeuroML2 document

        Parameters
        ----------
        ratefn : NeuroML element for rate function
            Element defining the rate function
        ctype: NeuroML ComponentType
            ComponentType matching that of the rate function
        vtab : sequence
            Sequence of values at which the rate has to be computed
        vshift: str
            Voltage shift to be applied (value with unit)
        param_tabs: dict {varname: (xp, fp)}
            Mapping variable name to interpolation table (see
            `calculateRateFn`)
        """
        rate = []
        params = None
        req_vars = {
            "vShift": vshift,
            "temperature": self._getTemperature(),
        }
        req_vars.update(self._variables)  # what's the use?
        if _isConcDep(ctype):
            caConcName = _findCaConcVariableName()  # moose CaCon element name
            req_vars["v"] = "0.0V"
        for vv in vtab:
            if param_tabs is not None:
                params = {
                    name: np.interp(vv, *tab)
                    for name, tab in param_tabs.items()
                }
            if _isConcDep(ctype):
                req_vars["caConc"] = f"{max(1e-11,vv):g}"  # do we need this?
                req_vars[caConcName] = f"{max(1e-11,vv):g}"
            else:
                req_vars["v"] = f"{vv}V"
            vals = pynml.evaluate_component(
                ctype, req_variables=req_vars, parameter_values=params
            )
            v = vals.get("x", vals.get("t", vals.get("r", None)))
            if v is not None:
                rate.append(v)
        return np.r_[rate]

    def calculateRateFn(
        self, ratefn, vmin, vmax, tablen=3000, vShift="0mV", param_tabs=None
    ):
        """Compute rate function based on NeuroML description.

        Calculate the HHGate rate functions from NeuroML
        description. In the simplest case it is forward or backward
        rate (alpha and beta) in one of the standard forms. But in
        some cases there needs to be further tweaks where the time
        course (tau) and steady state (inf) values are computed based
        on rules on alpha and beta.

        Parameters
        ----------
        ratefn : str
            NeuroML element describing HHRate
        vmin : Lower end of rate table
            Upper end of rate table
        vmax : str
            Upper end of rate table
        tablen : int
            Length of the table
        vShift : str
            Voltage shift to be applied
        param_tables : dict {variable_name: interpolation_table}
            Interpolation tables for gate parameters. The keys are used
            as variable names when evaluating neuroml dynamics expression.
            The values are tuples `(xp, fp)` where `xp` are the
            x-coordinates of the datapoints and `fp` are the y-coordinates.
            See ``numpy.interp`` for details of the latter.


        """
        vtab = np.linspace(vmin, vmax, tablen)

        # For standard rate functions in `PREDEFINED_RATEFN_MAP` simply call it
        # and return the result
        function = PREDEFINED_RATEFN_MAP.get(ratefn.type, None)
        if function is not None:
            midpoint = SI(ratefn.midpoint)
            rate = SI(ratefn.rate)
            scale = SI(ratefn.scale)
            return function(vtab, rate, scale, midpoint)

        # Non-standard rate function: find a ComponentType matching
        # the `type` of the rate function
        for ct in self.doc.ComponentType:
            if ratefn.type != ct.name:
                continue

            logger_.info(f"Using {ct.name} to evaluate rate")
            rate = self.rateFnFromFormula(
                ratefn, ct, vtab, vshift=vShift, param_tabs=param_tabs
            )
            return rate

        print(
            "[WARN ] Could not determine rate: %s %s %s"
            % (ratefn.type, vmin, vmax)
        )
        return np.empty(0)

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
                logger_.info("No channel with id: %s" % chdens.ion_channel)
                continue

            if self.verbose:
                logger_.info(
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
                        chdens,
                        self.nml_segs_to_moose[seg.id],
                        condDensity,
                        erev,
                    )
            """moose.le(self.nml_to_moose[seg])
            moose.showfield(self.nml_to_moose[seg], field="*", showtype=True)"""

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
                f"No prototype channel for {chdens.ion_channel}"
                f"referred to by {chdens.id}"
            )

        if self.verbose:
            logger_.info(
                f"Copying {chdens.id} to {comp}, {condDensity}; erev={erev}"
            )
        orig = chdens.id
        chid = moose.copy(proto_chan, comp, chdens.id)
        chan = moose.element(chid)
        chan.Gbar = sarea(comp) * condDensity
        chan.Ek = erev
        moose.connect(chan, "channel", comp, "channel")
        return chan

    def _is_standard_nml_rate(self, rate):
        return (
            rate.type == "HHExpLinearRate"
            or rate.type == "HHExpRate"
            or rate.type == "HHSigmoidRate"
            or rate.type == "HHSigmoidVariable"
        )

    def _computeQ10Scale(self, ngate):
        """Compute Q10 scale factor for NeuroML2 gate `ngate`.

        Parameters
        ----------
        ngate : NeuroML2 element
            Gate element in NeuroML2

        """
        q10_scale = 1.0
        if ngate.q10_settings:
            if ngate.q10_settings.type == "q10Fixed":
                q10_scale = float(ngate.q10_settings.fixed_q10)
            elif ngate.q10_settings.type == "q10ExpTemp":
                exp_temp = SI(ngate.q10_settings.experimental_temp)
                dtemp = self._getTemperature() - exp_temp
                q10_scale = math.pow(
                    float(ngate.q10_settings.q10_factor), dtemp / 10
                )
            else:
                raise Exception(
                    f"Unknown Q10 scaling type {ngate.q10_settings.type} "
                    f": {ngate.q10_settings}"
                )
        return q10_scale

    def updateHHGate(self, ngate, mgate, mchan, vmin, vmax, vdivs):
        """Update moose `HHGate` mgate from NeuroML gate description
        element `ngate`.

        Some neuroml descriptions work in two stages. Traditional
        steady state (minf/hinf) and time course (taum, tauh) are
        computed using a standard formula first, and then these are
        tweaked based on custom rules. This function updates a moose
        HHGate's tables by computing these as required.

        Parameters
        ----------
        ngate : neuroml element
            NeuroML gate description to be implemented
        mgate : HHGate
            Moose HHGate object to be updated
        mchan : HHChannel
            Moose HHCHannel object whose part the `mgate` is
        vmin : str
            minimum voltage (or concentration) for gate interpolation tables
        vmax : str
            Maximum voltage (or concentration) for gate interpolation
            tables
        vdivs : str
            Number of divisions in gate interpolation tables

        """
        # Set the moose gate powers from nml gate instance count
        if mgate.name.endswith("X"):
            mchan.Xpower = ngate.instances
        elif mgate.name.endswith("Y"):
            mchan.Ypower = ngate.instances
        elif mgate.name.endswith("Z"):
            mchan.Zpower = ngate.instances
        mgate.min = vmin
        mgate.max = vmax
        mgate.divs = vdivs
        q10_scale = self._computeQ10Scale(ngate)
        alpha, beta, tau, inf = (None, None, None, None)
        # First try computing alpha and beta from fwd and rev rate
        # specs. Set the gate tables using alpha and beta by default.
        fwd = ngate.forward_rate
        rev = ngate.reverse_rate
        param_tabs = None
        if (fwd is not None) and (rev is not None):
            alpha = self.calculateRateFn(
                fwd, vmin=vmin, vmax=vmax, tablen=vdivs
            )
            beta = self.calculateRateFn(rev, vmin=vmin, vmax=vmax, tablen=vdivs)
            vtab = np.linspace(vmin, vmax, vdivs)
            param_tabs = {"alpha": (vtab, alpha), "beta": (vtab, beta)}
            mgate.tableA = q10_scale * (alpha)
            mgate.tableB = q10_scale * (alpha + beta)
        # A `timeCourse` element indicates tau is not #
        # straightforward 1/(alpha+beta) but tweaked from alpha
        # and beta computed above
        if getattr(ngate, "time_course", None) is not None:
            tau = self.calculateRateFn(
                ngate.time_course,
                vmin=vmin,
                vmax=vmax,
                tablen=vdivs,
                param_tabs=param_tabs,
            )
        elif (alpha is not None) and (beta is not None):
            tau = 1.0 / (alpha + beta)
        # A `steadyState` element indicates inf is not
        # straightforward, but tweaked from alpha and beta computed
        # above
        if getattr(ngate, "steady_state", None) is not None:
            inf = self.calculateRateFn(
                ngate.steady_state,
                vmin=vmin,
                vmax=vmax,
                tablen=vdivs,
                param_tabs=param_tabs,
            )
        elif (alpha is not None) and (beta is not None):
            inf = alpha / (alpha + beta)
        # Update the gate tables only if tau or inf were tweaked
        if (tau is not None) and (inf is not None):
            mgate.tableA = q10_scale * inf / tau
            mgate.tableB = q10_scale * 1.0 / tau
        return mgate

    def createHHChannel(self, chan, vmin=-150e-3, vmax=100e-3, vdivs=5000):
        mchan = moose.HHChannel(f"{self.lib.path}/{chan.id}")
        mgates = [
            moose.element(x) for x in [mchan.gateX, mchan.gateY, mchan.gateZ]
        ]
        assert (
            len(chan.gate_hh_rates) <= 3
        ), "HHChannel allows only up to 3 gates"

        if self.verbose:
            logger_.info(
                "== Creating channel: "
                f"{chan.id} ({chan.gates}, {chan.gate_hh_rates})"
                f"-> {mchan} ({mgates})"
            )

        # Sort all_gates such that they come in x, y, z order.
        all_gates = _gates_sorted(chan.gates + chan.gate_hh_rates)
        for ngate, mgate in zip(all_gates, mgates):
            if ngate is None:
                continue
            mgate = self.updateHHGate(
                ngate=ngate,
                mgate=mgate,
                mchan=mchan,
                vmin=vmin,
                vmax=vmax,
                vdivs=vdivs,
            )
            logger_.debug(f"Updated HHGate {mgate.path}")
        logger_.info(
            f"{self.filename}: Created moose HHChannel {mchan.path} for neuroml {chan.id}"
        )
        return mchan

    def createPassiveChannel(self, chan):
        epath = f"{self.lib.path}/{chan.id}"
        if moose.exists(epath):
            mchan = moose.element(epath)
        else:
            mchan = moose.Leakage(epath)
        logger_.info(
            f"{self.filename}: created {mchan.path} for {chan.id}"
        )
        return mchan

    def importInputs(self, doc):
        """Create inputs to the model.

        Create PulseGen objects to deliver current injection. This is
        for convenience only, and may be overridden in future for more
        suitable experimental instrumentation/protocol description specs.

        Parameters
        ----------
        doc : NeuroMLDocument
            Document object read from NeuroML file

        """
        minputs = moose.Neutral(f"{self.lib.path}/inputs")
        for pg_nml in doc.pulse_generators:
            epath = f"{minputs.path}/{pg_nml.id}"
            pg = moose.PulseGen(epath)
            pg.firstDelay = SI(pg_nml.delay)
            pg.firstWidth = SI(pg_nml.duration)
            pg.firstLevel = SI(pg_nml.amplitude)
            pg.secondDelay = 1e9
            logger_.debug(f'{"$" * 10} Created input {epath}')

    def importIonChannels(self, doc, vmin=-150e-3, vmax=100e-3, vdivs=5000):
        logger_.info(f"{self.filename}: Importing the ion channels")

        for chan in doc.ion_channel + doc.ion_channel_hhs:
            if self.isPassiveChan(chan):
                mchan = self.createPassiveChannel(chan)
            else:
                mchan = self.createHHChannel(chan)

            self.id_to_ionChannel[chan.id] = chan
            self.nml_chans_to_moose[chan.id] = mchan
            self.proto_chans[chan.id] = mchan
            logger_.info(
                f"{self.filename}: Created ion channel {mchan.path} "
                f"for {chan.type} {chan.id}"
            )

    def importConcentrationModels(self, doc):
        for concModel in doc.decaying_pool_concentration_models:
            self.createDecayingPoolConcentrationModel(concModel)

    def createDecayingPoolConcentrationModel(self, concModel):
        """Create prototype for concentration model"""
        if hasattr(concModel, "name") and concModel.name is not None:
            name = concModel.name
        else:
            name = concModel.id

        ca = moose.CaConc(f"{self.lib.path}/{name}")
        ca.CaBasal = SI(concModel.resting_conc)
        ca.tau = SI(concModel.decay_constant)
        ca.thick = SI(concModel.shell_thickness)
        ca.B = 5.2e-6  # B = 5.2e-6/(Ad) where A is the area of the
        # shell and d is thickness - must divide by
        # shell volume when copying
        self.proto_pools[concModel.id] = ca
        self.nml_conc_to_moose[concModel.id] = ca
        self.moose_to_nml[ca] = concModel
        logger_.debug(
            f"Created moose element {ca.path} for nml {concModel.id}"
        )

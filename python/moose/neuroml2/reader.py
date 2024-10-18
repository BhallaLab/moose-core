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

import ast
import os
import math
import logging
import numpy as np
from collections import defaultdict
import pint

import moose
from moose.neuroml2.hhfit import exponential2
from moose.neuroml2.hhfit import sigmoid2
from moose.neuroml2.hhfit import linoid2
from moose.neuroml2.units import SI


logger_ = logging.getLogger("moose.nml2")
# logger_.setLevel(logging.DEBUG)
ureg = pint.UnitRegistry()
ureg.default_system = "SI"
Q_ = ureg.Quantity
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


PREDEFINED_RATEFN_MAP = {
    "HHExpRate": exponential2,
    "HHSigmoidRate": sigmoid2,
    "HHSigmoidVariable": sigmoid2,
    "HHExpLinearRate": linoid2,
}


def array_eval_component(comp_type, req_vars, params={}):
    """Use numpy vectorization for faster evaluation of component formula.

    Parameters
    ----------
    comp_type : nml.ComponentType
        ComponentType element defining the dynamics.
    req_vars : dict
        Variable names mapped to a pint Quantity (unit-aware number/array).
    params : dict
        Mapping from parameter names to a pint Quantity (unit-aware
        number/array). These are all passed to numpy, so they must be broadcast
        compatible.

    """
    logger_.debug(
        f"Evaluating {comp_type.name} with "
        f"req: {req_vars.keys()} and "
        f"parameters: {params}"
    )
    local_vars = {"return_vals": {}}
    for name, quantity in req_vars.items():
        local_vars[name] = quantity.to_base_units().magnitude
    if params is not None:
        for name, quantity in params.items():
            local_vars[name] = quantity.to_base_units().magnitude
    exec_str = []
    for const in comp_type.Constant:
        exec_str.append(f"{const.name} = {pynml.get_value_in_si(const.value)}")
    for dyn in comp_type.Dynamics:
        for dv in dyn.DerivedVariable:
            exec_str.append(f"{dv.name} = {dv.value}")
            exec_str.append(f"return_vals['{dv.name}'] = {dv.name}")
        for cdv in dyn.ConditionalDerivedVariable:
            cond_str = [f"return_vals['{cdv.name}'] = "]
            closing_parens = 0
            for case_ in cdv.Case:
                if case_.condition is not None:
                    cond = (
                        case_.condition.replace(".neq.", "!=")
                        .replace(".eq.", "==")
                        .replace(".gt.", ">")
                        .replace(".lt.", "<")
                    )
                    cond_str.append(f"where({cond}, {case_.value}, ")
                    closing_parens += 1
                else:
                    cond_str += [case_.value, ")" * closing_parens]
            # print("^" * 100, cond_str)
            exec_str.append(" ".join(cond_str))
    # print("#" * 80, "\n", exec_str, "\n", "#" * 80, "\n")
    exec_str = "\n".join(exec_str)
    # print("*" * 80, "\n", exec_str, "\n", "*" * 80)
    exec(exec_str, np.__dict__, local_vars)
    return local_vars["return_vals"]


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
    If the id of gates are subset of 'x', 'y' or 'z' then sort them so they
    load in X, Y or Z gate respectively. Otherwise do not touch them i.e.
    first gate will be loaded into X, second into Y and so on.
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
    if len(caConcs) >= 1:
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
    """
    Return the cross sectional area (pi * r^2) from the diameter of the
    compartment.

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
    return list(set(segments))


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

    def read(
        self,
        filename,
        symmetric=True,
        vmin=-150e-3,
        vmax=100e-3,
        vdivs=3000,
        cmin=0.0,
        cmax=10.0,
        cdivs=5000,
    ):
        """Read a NeuroML2 file and create the corresponding model.

        Parameters
        ----------
        filename : str
            Path of NeuroML2 file
        symmetric : bool
            If `True` use symmetric compartments (axial resistance is
            split with neighboring compartments on both sides)
            otherwise asymmetric compartment (axial resistance applied
            with one neighboring compartment only)
        vmin : float
            Minimum of membrane voltage range. This is used for
            creating interpolation tables for channel dynamics.
        vmax : float
            Maximum of membrane voltage range. This is used for
            creating interpolation tables for channel dynamics.
        vdivs : int
            Number of entries in voltage range for interpolation. This is
            used for creating interpolation tables for channel dynamics.
        cmin : float
            Minimum (Ca2+) concentration
        cmax : float
            Maximum (Ca2+) concentration
        cdivs : int
            Number of entries in concentration range for interpolation. This is
            used for creating interpolation tables for [Ca2+]-dependent channel
            dynamics.


        """
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
        self.importIonChannels(
            self.doc,
            vmin=vmin,
            vmax=vmax,
            vdivs=vdivs,
            cmin=cmin,
            cmax=cmax,
            cdivs=cdivs,
        )
        self.importInputs(self.doc)

        for cell in self.doc.cells:
            self.createCellPrototype(cell, symmetric=symmetric)

        if len(self.doc.networks) >= 1:
            self.createPopulations()
            self.createInputs()

    def _getTemperature(self):
        """Get the network temperature.

        If there is no network attribute, or if the network has no
        temperature attribute, return standard room temperature

        """
        try:
            return SI(self.network.temperature)
        except AttributeError:
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
            f"{self.model.path}/{self.network.id}/"
            f"{pop_id}/{cellIndex}/{comp_name}"
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
        """Create the morphology, channels in prototype.

        Parameters
        ----------
        cell: NeuroML element
            Cell element in NeuroML2
        symmetric: bool
            If `True`, use symmetric compartment; use asymmetric compartment
            otherwise.

        """
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

        Parameters
        ----------
        nmlcell: NeuroML element
            Cell element in NeuroML2
        moosecell: moose.Neutral or moose.Neuron
            MOOSE container object for the cell.
        symmetric: bool
            If `True`, use symmetric compartment; use asymmetric compartment
            otherwise.

        NOTE
        ----
        moose compartments are cylindrical (both ends of a compartment have
        same diameter). So taking the average of the two ends in case of truncated-cone.

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
        self.setupCaDep(nmlcell, moosecell, bp.membrane_properties)

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
        """Copy ion species prototype from /library to segments in the cell.

        Handling [Ca2+]: NeuroML2 `decayingPoolConcentrationModel` for
        Ca2+ ion is mapped to `CaConc` class in MOOSE. However, while
        NeuroML2 allows setting `initialConcentration` for this under
        `intracellularProperties` in the cell model, MOOSE does not
        have a separate initial value field for the
        concentration. Instead, the
        `decayingPoolConcentrationModel.restingConc` is used for
        setting `CaConc.CaBasal` and the pool is initialized to this
        value.

        """
        sg_to_segments = self._cell_to_sg[nmlcell.id]
        for species in properties.species:
            proto_pool = None
            concModel = species.concentration_model
            if concModel is None:
                continue
            if concModel in self.proto_pools:
                proto_pool = self.proto_pools[concModel]
            else:
                for innerReader in self.includes.values():
                    if concModel in innerReader.proto_pools:
                        proto_pool = innerReader.proto_pools[concModel]
                        break
            if proto_pool is None:
                msg = "No prototype pool for %s referred to by %s" % (
                    concModel,
                    species.id,
                )
                logger_.error(msg)
                raise RuntimeError(msg)
            segments = getSegments(nmlcell, species, sg_to_segments)
            for seg in segments:
                comp = self.nml_segs_to_moose[seg.id]
                self.copySpecies(proto_pool, comp)
                # moose initializes CaConc to `CaBasal` which
                # is equivalent to `restingConc` in NeuroML2

    def copySpecies(self, proto_pool, comp):
        """Copy the prototype pool `species` to compartment
        `comp`. Currently only decaying pool of Ca2+ supported.

        """
        pool_id = moose.copy(proto_pool, comp)
        pool = moose.element(pool_id)
        pool.diameter = comp.diameter
        pool.length = comp.length
        # for cylindrical compartments (default) moose computes volume
        # and pool.B by standard formula, but spherical case is not
        # implemented in CaConBase
        if comp.length <= 0:
            vol = (
                4
                * np.pi
                * (
                    (0.5 * comp.diameter) ** 3
                    - (0.5 * comp.diameter - pool.thick) ** 3
                )
                / 3
            )
            pool.B = 5.2e-6 / vol

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

    def getComponentType(self, ratefn):
        """Returns the NeuroML ComponentType object for the ratefn.

        A NeuroMLDocument object has a list called 'ComponentType' of
        `nml.ComponentType` objects containing the ComponentType
        definitions. The rate equations for ion channel gate dynamics
        are also defined as `ComponentType` elements. The
        `ComponentType` object whose `name` matches the rate
        function's `type` attribute is returned.

        Parameters
        ----------
        ratefn : nml.HHRate
            Rate function element

        """
        for ct in self.doc.ComponentType:
            if ratefn.type == ct.name:
                return ct

    def rateFnFromFormula(
        self, ratefn, ctype, vtab, ctab=None, param_tabs={}, vshift="0.0V"
    ):
        """Compute rate values from formula provided in NeuroML2 document

        Parameters
        ----------
        ratefn : NeuroML element for rate function
            Element defining the rate function
        ctype: NeuroML ComponentType
            ComponentType matching that of the rate function
        vtab : Sequence
            Sequence of values at which the rate has to be computed.
            This is the primary parameter. Even if the rate is concentration
            dependent, but not voltage dependent, the array of concentration
            values must be passed via `vtab`, as it only requires a 1D
            interpolation table.
        ctab : Sequence
            Sequence of values for a second parameter like concentration.
            When this is specified, the rates are calculated for a 2D
            interpolation table of len(vtab)xlen(ctab), which is applicable
            for HHGate2D.
        param_tabs: dict {varname: fp}
            Mapping variable name to precomputed value table (see
            `calculateRateFn` for details)
        vshift: str
            Voltage shift to be applied (value with unit)
        """
        assert (vtab is not None) or (
            ctab is not None
        ), "At least one of voltage and concentration arrays must be specified"
        rate = []
        req_vars = {
            "vShift": Q_(vshift),
            "temperature": Q_(self._getTemperature()),
        }
        req_vars.update(self._variables)  # what's the use?
        if (vtab is None) or (len(vtab) == 0):
            req_vars["caConc"] = Q_(ctab, "mole / meter ** 3")
        elif (ctab is None) or (len(ctab) == 0):
            req_vars["v"] = Q_(vtab, "V")
        else:
            # Get pair-wise coordinates for 2D table
            vv, cc = np.meshgrid(vtab, ctab)
            req_vars["v"] = Q_(vv.flatten(), "V")
            req_vars["caConc"] = Q_(cc.flatten(), "mole / meter ** 3")

        vals = array_eval_component(
            ctype, req_vars=req_vars, params=param_tabs
        )
        rate = vals.get("x", vals.get("t", vals.get("r", None)))
        if rate is None:
            raise ValueError("Evaluation of expression returned None")
        if (vtab is not None) and (ctab is not None):
            # Transpose meshgrid which creates m x n array from n-vec and
            # m-vec. MOOSE table assumes n x m lookup table where n is
            # first index variable array is of length n and the second
            # index variable array if of length m
            return rate.reshape(len(ctab), len(vtab)).T
        return rate

    def calculateRateFn(
        self, ratefn, vtab, ctab=None, param_tabs={}, vShift="0mV"
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
        vtab: Sequence
            Array of voltage/concentration values for which the rate
            function is computed
        ctab : Sequence (default: None)
            Sequence of values for a parameter like concentration.
            When  both `vtab` and `ctab` are specified, the rates are
            calculated for a 2D interpolation table of len(vtab)xlen(ctab),
            which is applicable for HHGate2D.
        param_tabs : dict {variable_name: value_array}
            Precomputed tables for gate parameters. The keys are used
            as variable names when evaluating neuroml dynamics expression.
            The values are arrays (or sequences) with entries
            corresponding to `vtab`.
        vShift : str
            Voltage shift to be applied


        """
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
        ct = self.getComponentType(ratefn)
        logger_.info(f"Using {ct.name} to evaluate rate")
        rate = self.rateFnFromFormula(
            ratefn,
            ct,
            vtab,
            ctab=ctab,
            param_tabs=param_tabs,
            vshift=vShift,
        )
        if rate is None:
            print(f"[WARN ] Could not determine rate: {ratefn.type}")
            return np.empty(0)
        return rate

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
                    f"Setting density of channel {chdens.id} in {segments}"
                    f"to {condDensity}; erev={erev} "
                    f"(passive: {self.isPassiveChan(ionChannel)})"
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

    def isDynamicsCaDependent(self, ct):
        """Returns True if the dynamics of `ct` is Ca dependent.

        If any identifier in the value expression of a DerivedVariable
        or ConditionalDerivedVariable in the Dynamics of this
        ComponentType is `caConc`, we assume it is [Ca2+] dependent.

        Parameters
        ----------
        ct : nml.ComponentType
            ComponentType object

        """
        for dyn in ct.Dynamics:
            for dv in dyn.DerivedVariable + dyn.ConditionalDerivedVariable:
                parsed = ast.parse(dv.value)
                # If any identifier (represented by class ast.Name) in
                # the derived variable expression is "caConc", the
                # gate is Ca concentration dependent.
                for node in ast.walk(parsed):
                    if isinstance(node, ast.Name):
                        if node.id == "caConc":
                            return True
        return False

    def isDynamicsVoltageDependent(self, ct):
        """Returns True if the dynamics of `ct` is Voltage dependent.

        If any identifier in the value expression of a DerivedVariable
        or ConditionalDerivedVariable in the Dynamics of this
        ComponentType is `v`, we assume it is voltage dependent.

        Parameters
        ----------
        ct : nml.ComponentType
            ComponentType object

        """
        for dyn in ct.Dynamics:
            for dv in dyn.DerivedVariable + dyn.ConditionalDerivedVariable:
                parsed = ast.parse(dv.value)
                # If any identifier (represented by class ast.Name) in
                # the derived variable expression is "caConc", the
                # gate is Ca concentration dependent.
                for node in ast.walk(parsed):
                    if isinstance(node, ast.Name):
                        if node.id == "v":
                            return True
        return False

    def isDynamicsVoltageCaDependent(self, ct):
        """Returns True if the dynamics of `ct` is dependent on both
        voltage and calcium concentration.

        Identifiers `caConc` and `v` both appear in the value
        expressions of the DerivedVariable or
        ConditionalDerivedVariable elements in the Dynamics of this
        ComponentType, we assume it is [Ca2+] and voltage dependent.

        Gates whose dynamics depend on both voltage and Ca2+
        concentration require a 2D interpolation table, and are
        implemented as HHGate2D under HHChannel2D in moose. This test
        is need for that.

        Parameters
        ----------
        ct : nml.ComponentType
            ComponentType object

        """
        ca_dep = False
        v_dep = False
        for dyn in ct.Dynamics:
            for dv in dyn.DerivedVariable + dyn.ConditionalDerivedVariable:
                parsed = ast.parse(dv.value)
                # If any identifier (represented by class ast.Name) in
                # the derived variable expression is "caConc", the
                # gate is Ca concentration dependent. If any
                # identifier is "v", it is voltage dependent.
                for node in ast.walk(parsed):
                    if isinstance(node, ast.Name):
                        if node.id == "caConc":
                            ca_dep = True
                        if node.id == "v":
                            v_dep = True
        return (ca_dep is True) and (v_dep is True)

    def isChannelCaDependent(self, chan):
        """Returns True if `chan` is dependent on calcium concentration.

        Checks if any of the gate dynamics depend on Ca
        concentration. We assumme that if the ComponentType for the
        gate has a Dynamics element where any of the rates (neuroml
        DerivedVariable) contain a value `caConc`.

        Parameters
        ----------
        chan : nml.IonChannel
            NeuroML IonChannel element

        """
        for gate in chan.gates + chan.gate_hh_rates:
            dynamics = [
                gate.forward_rate,
                gate.reverse_rate,
                gate.time_course,
                gate.steady_state,
            ]
            for dyn in dynamics:
                if dyn is not None:
                    ct = self.getComponentType(dyn)
                    if ct is None or (ct.extends != "baseVoltageConcDepRate"):
                        continue
                    if self.isDynamicsCaDependent(ct):
                        return True
        return False

    def isChannel2D(self, chan):
        """Returns True if `chan` requires moose.HHChannel2D.

        For channels where any of the gates is dependent on both [Ca2+]
        and voltage, the gate opening/closing rates are looked up from a
        2D interpolation table indexed by both voltage and
        concentration. This is implemented in HHGate2D. The channel is
        implemented as HHChannel2D allows up to HHGate2D.

        Parameters
        ----------
        chan: nml.IonChannel
            IonChannel element to be checked.
        """
        for gate in chan.gates + chan.gate_hh_rates:
            if self.isGateVoltageCaDependent(gate):
                return True
        return False

    def isGateCaDependent(self, ngate):
        """Returns True if `ngate` has dynamics that depends on Ca2+
        concentration.

        Parameters
        ----------
        ngate : nml.HHGate
            Gate description element in neuroml

        """
        dynamics = [
            getattr(ngate, "forward_rate", None),
            getattr(ngate, "reverse_rate", None),
            getattr(ngate, "time_course", None),
            getattr(ngate, "steady_state", None),
        ]
        for dyn in dynamics:
            if dyn is not None:
                ct = self.getComponentType(dyn)
                if (ct is not None) and self.isDynamicsCaDependent(ct):
                    return True
        return False

    def isGateVoltageDependent(self, ngate):
        """Returns True if `ngate` has dynamics that depends voltage.

        Parameters
        ----------
        ngate : nml.HHGate
            Gate description element in neuroml

        """
        dynamics = [
            getattr(ngate, "forward_rate", None),
            getattr(ngate, "reverse_rate", None),
            getattr(ngate, "time_course", None),
            getattr(ngate, "steady_state", None),
        ]
        for dyn in dynamics:
            if dyn is not None:
                ct = self.getComponentType(dyn)
                if (ct is not None) and self.isDynamicsCaDependent(ct):
                    return True
        return False

    def isGateVoltageCaDependent(self, ngate):
        """Returns True if `ngate` has dynamics that depends on both voltage
        and Ca2+ concentration.

        Parameters
        ----------
        ngate : nml.HHGate
            Gate description element in neuroml

        """
        dynamics = [
            getattr(ngate, "forward_rate", None),
            getattr(ngate, "reverse_rate", None),
            getattr(ngate, "time_course", None),
            getattr(ngate, "steady_state", None),
        ]
        for dyn in dynamics:
            if dyn is not None:
                ct = self.getComponentType(dyn)
                if (
                    (ct is not None)
                    and (ct.extends == "baseVoltageConcDepRate")
                    and self.isDynamicsVoltageCaDependent(ct)
                ):
                    return True
        return False

    def setupCaDep(self, nmlcell, moosecell, membrane_properties):
        """Create connections between Ca channels, Ca pool, and
        Ca-dependent channels."""
        sg_to_segments = self._cell_to_sg[nmlcell.id]
        caName = _findCaConcVariableName()
        if caName is None:
            print(
                "[INFO ] No CaConc object found. Skip setting up [Ca2+]"
                " dependence."
            )
            return
        for chdens in (
            membrane_properties.channel_densities
            + membrane_properties.channel_density_v_shifts
        ):
            segments = getSegments(nmlcell, chdens, sg_to_segments)
            try:
                nchan = self.id_to_ionChannel[chdens.ion_channel]
            except KeyError:
                logger_.info("No channel with id: %s" % chdens.ion_channel)
                continue
            if chdens.ion == "ca":
                for seg in segments:
                    comp = self.nml_segs_to_moose[seg.id]
                    caPool = moose.element(f"{comp.path}/{caName}")
                    mchan = moose.element(f"{comp.path}/{chdens.id}")
                    if caPool.path != "/":
                        moose.connect(mchan, "IkOut", caPool, "current")
            if self.isChannelCaDependent(nchan):
                for seg in segments:
                    comp = self.nml_segs_to_moose[seg.id]
                    mchan = moose.element(f"{comp.path}/{chdens.id}")
                    caPool = moose.element(f"{comp.path}/{caName}")
                    if caPool.path != "/":
                        moose.connect(caPool, "concOut", mchan, "concen")

    def copyChannel(self, chdens, comp, condDensity, erev):
        """Copy moose prototype for `chdens` condutcance density to `comp`
        compartment.

        """
        proto_chan = None
        try:
            proto_chan = self.proto_chans[chdens.ion_channel]
        except KeyError:
            for innerReader in self.includes.values():
                try:
                    proto_chan = innerReader.proto_chans[chdens.ion_channel]
                    break
                except KeyError:
                    pass
        if not proto_chan:
            raise Exception(
                f"No prototype channel for {chdens.ion_channel}"
                f"referred to by {chdens.id}"
            )

        if self.verbose:
            logger_.info(
                f"Copying {chdens.id} to {comp}, {condDensity}; erev={erev}"
            )
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

    def updateHHGate(self, ngate, mgate, mchan, vmin, vmax, vdivs, useInterpolation=True):
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
        mgate.useInterpolation = useInterpolation
        q10_scale = self._computeQ10Scale(ngate)
        alpha, beta, tau, inf = (None, None, None, None)
        param_tabs = {}
        vtab = np.linspace(vmin, vmax, vdivs)
        # First try computing alpha and beta from fwd and rev rate
        # specs. Set the gate tables using alpha and beta by default.
        fwd = ngate.forward_rate
        rev = ngate.reverse_rate
        if (fwd is not None) and (rev is not None):
            alpha = self.calculateRateFn(fwd, vtab)
            beta = self.calculateRateFn(rev, vtab)
            param_tabs = {"alpha": Q_(alpha, "1/s"), "beta": Q_(beta, "1/s")}

        # Beware of a peculiar cascade of evaluation below: In some
        # cases rate parameters alpha and beta are computed with
        # standard HH-type formula, then tweaked based on the
        # existence of "timeCourse" or "steadyState" attributes in the
        # channel definition. The results depend on the order of
        # execution.

        # A `timeCourse` element indicates tau is not #
        # straightforward 1/(alpha+beta) but tweaked from alpha
        # and beta computed above
        if getattr(ngate, "time_course", None) is not None:
            tau = self.calculateRateFn(
                ngate.time_course,
                vtab,
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
                vtab,
                param_tabs=param_tabs,
            )
        elif (alpha is not None) and (beta is not None):
            inf = alpha / (alpha + beta)

        # Should update the gate tables only if `tau` or `inf` were
        # tweaked, but this is simpler than checking the cascading
        # evaluation above.
        mgate.tableA = q10_scale * inf / tau
        mgate.tableB = q10_scale * 1.0 / tau
        # DEBUG
        # np.savetxt(
        #     f"{mchan.name}.{ngate.id}.inf.txt",
        #     np.block([vtab[:, np.newaxis], inf[:, np.newaxis]]),
        # )
        # np.savetxt(
        #     f"{mchan.name}.{ngate.id}.tau.txt",
        #     np.block([vtab[:, np.newaxis], inf[:, np.newaxis]]),
        # )
        # import matplotlib.pyplot as plt

        # fig, ax = plt.subplots(nrows=1, ncols=2)
        # ax[0].plot(vtab, inf)
        # ax[1].plot(vtab, tau)
        # fig.suptitle(f"{mchan.name}/{ngate.id}")
        # END DEBUG
        return mgate

    def updateHHGate2D(
        self, ngate, mgate, mchan, vmin, vmax, vdivs, cmin, cmax, cdivs
    ):
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
        cmin : float
            Minimum concentration
        cmax : float
            Maximum concentration
        cdivs : int
            Number of divisions in the interpolation table for concentration

        """
        # Set the moose gate powers from nml gate instance count
        if mgate.name.endswith("X"):
            mchan.Xpower = ngate.instances
        elif mgate.name.endswith("Y"):
            mchan.Ypower = ngate.instances
        elif mgate.name.endswith("Z"):
            mchan.Zpower = ngate.instances
        # TODO: HHGate2D and and HHChannel2D should be updated in
        # MOOSE to avoid this redundant setting for the two tables
        mgate.xmin = vmin
        mgate.xmax = vmax
        mgate.xdivs = vdivs
        mgate.ymin = cmin
        mgate.ymax = cmax
        mgate.ydivs = cdivs
        # If the gate depends only on one parameter, disable the
        # second dimension
        q10_scale = self._computeQ10Scale(ngate)
        alpha, beta, tau, inf = (None, None, None, None)
        param_tabs = {}
        vtab = np.linspace(vmin, vmax, vdivs)
        ctab = np.linspace(cmin, cmax, cdivs)
        # First try computing alpha and beta from fwd and rev rate
        # specs. Set the gate tables using alpha and beta by default.
        fwd = ngate.forward_rate
        rev = ngate.reverse_rate
        if (fwd is not None) and (rev is not None):
            alpha = self.calculateRateFn(fwd, vtab, ctab=ctab)
            beta = self.calculateRateFn(rev, vtab, ctab=ctab)
            param_tabs = {"alpha": Q_(alpha, "1/s"), "beta": Q_(beta, "1/s")}
        # Beware of a peculiar cascade of evaluation below: In some
        # cases rate parameters alpha and beta are computed with
        # standard HH-type formula, then tweaked based on the
        # existence of "timeCourse" or "steadyState" attributes in the
        # channel definition. The results depend on the order of
        # execution.

        # A `timeCourse` element indicates tau is not #
        # straightforward 1/(alpha+beta) but tweaked from alpha
        # and beta computed above
        if getattr(ngate, "time_course", None) is not None:
            tau = self.calculateRateFn(
                ngate.time_course,
                vtab,
                ctab=ctab,
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
                vtab,
                ctab=ctab,
                param_tabs=param_tabs,
            )
        elif (alpha is not None) and (beta is not None):
            inf = alpha / (alpha + beta)

        # Should update the gate tables only if `tau` or `inf` were
        # tweaked, but this is simpler than checking the cascading
        # evaluation above.
        mgate.tableA = q10_scale * inf / tau
        mgate.tableB = q10_scale * 1.0 / tau
        # DEBUG
        # if len(inf.shape) == 1:
        #     np.savetxt(
        #         f"{mchan.name}.{ngate.id}.inf.txt",
        #         np.block([vtab[:, np.newaxis], inf[:, np.newaxis]]),
        #     )
        #     np.savetxt(
        #         f"{mchan.name}.{ngate.id}.tau.txt",
        #         np.block([vtab[:, np.newaxis], inf[:, np.newaxis]]),
        #     )
        #     import matplotlib.pyplot as plt

        #     fig, ax = plt.subplots(nrows=1, ncols=2)
        #     ax[0].plot(vtab, inf)
        #     ax[1].plot(vtab, tau)
        #     fig.suptitle(f"{mchan.name}/{ngate.id}")
        # END DEBUG
        return mgate

    def createHHChannel(self, chan, vmin=-150e-3, vmax=100e-3, vdivs=3000):
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
            f"{self.filename}: Created moose HHChannel {mchan.path}"
            f" for neuroml {chan.id}"
        )
        return mchan

    def createHHChannel2D(
        self,
        chan,
        vmin=-150e-3,  # -150 mV
        vmax=100e-3,  # 100 mV
        vdivs=3000,
        cmin=0,
        cmax=10.0,
        cdivs=5000,
    ):
        """Create and return a Hodgkin-Huxley-type ion channel whose
        dynamics depends on 2-parameters.

        Creates a channel where some gate has dynamics defined by a
        non-separable function of Ca2+ concentration and voltage.

        Parameters
        ----------
        chan : nml.IonChannel
            NeuroML channel description
        vmin : float (default: -150 mV)
            Minimum voltage
        vmax : float (default: 100 mV)
            Maximum voltage
        vdivs : int
            Number of divisions in the interpolation table for voltage
        cmin : float (default: 0)
            Minimum concentration.
        cmax : float (default: 10.0 = 10 mM)
            Maximum concentration
        cdivs : int
            Number of divisions in the interpolation table for concentration

        """

        mchan = moose.HHChannel2D(f"{self.lib.path}/{chan.id}")
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
            # We need to update Xindex for gateX, Yindex for gateY and
            # Zindex for gateZ
            indexattr = f"{mgate.name[-1]}index"
            vdep = self.isGateCaDependent(ngate)
            cdep = self.isGateVoltageDependent(ngate)
            cmin_, cmax_, cdivs_ = 0, 0, 0
            vmin_, vmax_, vdivs_ = 0, 0, 0
            if vdep and cdep:
                # gate depends on both voltage and Ca - index for
                # first dimension is membrane potential and second
                # dimension is concentration
                setattr(mchan, indexattr, "VOLT_C1_INDEX")
                cmin_, cmax_, cdivs_ = cmin, cmax, cdivs
                vmin_, vmax_, vdivs_ = vmin, vmax, vdivs
            elif vdep and not cdep:
                setattr(mchan, indexattr, "VOLT_INDEX")
                vmin_, vmax_, vdivs_ = vmin, vmax, vdivs
            elif cdep and not vdep:
                setattr(mchan, indexattr, "C1_INDEX")
                cmin_, cmax_, cdivs_ = cmin, cmax, cdivs
            # Not handling a second concentration parameter concen2
            mgate = self.updateHHGate2D(
                ngate=ngate,
                mgate=mgate,
                mchan=mchan,
                vmin=vmin_,
                vmax=vmax_,
                vdivs=vdivs_,
                cmin=cmin_,
                cmax=cmax_,
                cdivs=cdivs_,
            )
            if not (vdep and cdep):
                mgate.ydivsA = 0
                mgate.ydivsB = 0

        logger_.info(
            f"{self.filename}: Created moose HHChannel {mchan.path}"
            f" for neuroml {chan.id}"
        )
        return mchan

    def createPassiveChannel(self, chan):
        epath = f"{self.lib.path}/{chan.id}"
        if moose.exists(epath):
            mchan = moose.element(epath)
        else:
            mchan = moose.Leakage(epath)
        logger_.info(f"{self.filename}: created {mchan.path} for {chan.id}")
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

    def importIonChannels(
        self,
        doc,
        vmin=-150e-3,
        vmax=100e-3,
        vdivs=3000,
        cmin=0,
        cmax=10.0,
        cdivs=5000,
    ):
        logger_.info(f"{self.filename}: Importing the ion channels")

        for chan in doc.ion_channel + doc.ion_channel_hhs:
            if self.isPassiveChan(chan):
                mchan = self.createPassiveChannel(chan)
            elif self.isChannel2D(chan):
                mchan = self.createHHChannel2D(
                    chan,
                    vmin=vmin,
                    vmax=vmax,
                    vdivs=vdivs,
                    cmin=cmin,
                    cmax=cmax,
                    cdivs=cdivs,
                )
            else:
                mchan = self.createHHChannel(
                    chan, vmin=vmin, vmax=vmax, vdivs=vdivs
                )

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
        """Create prototype for concentration model."""
        if hasattr(concModel, "name") and concModel.name is not None:
            name = concModel.name
        else:
            name = concModel.id
        ca = moose.CaConc(f"{self.lib.path}/{name}")
        ca.CaBasal = SI(concModel.resting_conc)
        ca.tau = SI(concModel.decay_constant)
        ca.thick = SI(concModel.shell_thickness)
        # Here is an implementation issue:
        # Decaying Ca pool is modeled as a thin shell of thickness d and
        # area A that decays towards the baseline/resting concentration
        # with a time constant tau:

        # d[Ca2+]/dt = I_Ca/(2 * F * A * d) - ([Ca2+] - [Ca2+]_rest)/tau

        # where F is Faraday's constant and I_Ca are the Ca currents
        # increasing the [Ca2+] in this shell:

        # rate of increase of the number of Ca2+ ions = I_Ca/2,

        # rate of increase of number of moles = I_Ca/(2 * F),

        # volume of shell = A * d,

        # rate of increase of concentration = I_Ca/(2 * F * A * d)

        # NeuroML puts the 1/(2 * F * d) factor as the attribute `rho`,
        # which may include some additional scaling.

        # In MOOSE implementation, setting B explicitly is
        # discouraged, but kept for backwards compatibility with
        # GENESIS. It is calculated dynamically using Faraday's
        # constant, valence, and volume of the shell. Every time the
        # volume is updated (upon setting thickness, length, or
        # diameter of the shell), it recomputes this factor.

        # The only problem is the calculation assumes cylindrical
        # shell, not accounting for spherical compartments.
        # - subha
        #
        # if concModel.rho is not None:
        #     # B = 5.2e-6/(Ad) where A is the area of the shell and d
        #     # is thickness - must divide by shell volume when copying
        #     ca.B = 5.2e-6

        # else:
        #     ca.B = Q_(concModel.rho).to_base_units().m
        self.proto_pools[concModel.id] = ca
        self.nml_conc_to_moose[concModel.id] = ca
        self.moose_to_nml[ca] = concModel
        logger_.debug(
            f"Created moose element {ca.path} for" f" nml {concModel.id}"
        )

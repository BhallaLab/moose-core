# test_granule98.py ---
#
# Filename: test_granule98.py
# Description:
# Author: Subhasis Ray
# Created: Mon Apr  8 21:41:22 2024 (+0530)
#           By: Subhasis Ray
#

# Code:
"""Test code for the Granule cell model

"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# import unittest
import logging


LOGLEVEL = os.environ.get("LOGLEVEL", "INFO")
logging.basicConfig(level=LOGLEVEL)


import moose
from moose.neuroml2.reader import NML2Reader


channels = ["NaF", "KDR", "KA", "KC"]


def dump_gate_tables(channel, do_plot=False):
    if do_plot:
        fig, axes = plt.subplots(nrows=1, ncols=2)

    for letter in ["X", "Y", "Z"]:
        power = getattr(channel, f"{letter}power")
        if power > 0:
            gate = moose.element(f"{channel.path}/gate{letter}")
            minf = gate.tableA / gate.tableB
            mtau = 1 / gate.tableB
            v = np.linspace(gate.min, gate.max, len(minf))
            inf_f = f"{channel.name}.{letter}.inf.dat"
            tau_f = f"{channel.name}.{letter}.tau.dat"
            np.savetxt(inf_f, np.block([v[:, np.newaxis], minf[:, np.newaxis]]))
            np.savetxt(tau_f, np.block([v[:, np.newaxis], mtau[:, np.newaxis]]))
            logging.info(
                f"Saved {letter} gate tables for {channel.name} in {inf_f} and {tau_f}"
            )
            if do_plot:
                axes[0].plot(v, minf, label=f"inf {letter}")
                axes[1].plot(v, mtau, label=f"tau {letter}")
    if do_plot:
        axes[0].legend()
        axes[1].legend()
        fig.suptitle(channel.name)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} modeldir")
        print(
            "where modeldir contains the neuroML2 model. "
            "For the GranuleCell model cloned "
            "from OpenSourceBrain, this should be "
            "`GranuleCell/neuroConstruct/generatedNeuroML2/`"
        )
        sys.exit(1)
    modeldir = sys.argv[1]
    do_plot = True
    reader = NML2Reader(verbose=True)
    filename = os.path.join(modeldir, "GranuleCell.net.nml")
    reader.read(filename)
    pop_id = reader.doc.networks[0].populations[0].id
    soma = reader.getComp(pop_id, cellIndex=0, segId=0)
    data = moose.Neutral("/data")
    pg = reader.getInput("Gran_10pA")
    pg.firstWidth = 500e-3
    inj = moose.Table(f"{data.path}/pulse")
    moose.connect(inj, "requestOut", pg, "getOutputValue")
    vm = moose.Table(f"{data.path}/Vm")
    moose.connect(vm, "requestOut", soma, "getVm")
    catab = None
    for el in moose.wildcardFind("/model/##[TYPE=CaConc]"):
        capool = moose.element(f"{soma.path}/{el.name}")
        catab = moose.Table(f"{data.path}/{capool.name}")
        moose.connect(catab, "requestOut", capool, "getCa")
        print("Recording", capool.path, ".Ca in", catab.path)

    chanmap = {}
    gktabs = []
    for el in moose.wildcardFind(
        f"{soma.path}/##[TYPE=HHChannel]"
    ) + moose.wildcardFind(f"{soma.path}/##[TYPE=HHChannel2D]"):
        chan = moose.element(el)
        chanmap[chan.name.split('_')[1]] = chan  # for debugging
        tab = moose.Table(f"{data.path}/{chan.name}")
        moose.connect(tab, "requestOut", chan, "getGk")
        gktabs.append(tab)
        print("Recording", chan.path, ".Gk in", tab.path)

    if catab is not None:
        gktabs.append(catab)

    for ch in soma.children:
        if moose.isinstance_(ch, moose.HHChannel):
            dump_gate_tables(ch, do_plot=do_plot)

    kca = chanmap['KCa']
    gate = moose.element(f'{kca.path}/gateX')
    vtab = np.linspace(gate.xmin, gate.xmax, gate.xdivs)
    ctab = np.linspace(gate.ymin, gate.ymax, gate.ydivs)
    cplot = [gate.A[-0.65, cc] for cc in ctab]
    vplot = [gate.A[vv, 7.55e-5] for vv in vtab]
    # fig, axes = plt.subplots(nrows=2)
    # axes[0].plot(ctab, cplot, 'x')
    # axes[0].set_xlabel('[Ca2+]')
    # axes[0].set_ylabel('A')
    # axes[1].plot(vtab, vplot, 'x')
    # axes[1].set_xlabel('V')
    # axes[1].set_ylabel('A')
    # plt.show()
    simtime = 700e-3
    moose.reinit()
    # breakpoint()
    moose.start(simtime)

    t = np.arange(len(vm.vector)) * vm.dt
    print("%" * 10, len(vm.vector), len(inj.vector))
    results = np.block(
        [t[:, np.newaxis], vm.vector[:, np.newaxis], inj.vector[:, np.newaxis]]
    )
    fname = "Granule_98.dat"
    np.savetxt(fname, X=results, header="time Vm Im", delimiter=" ")
    print(f"Saved time, Vm, Im, in {fname}")
    for tab in gktabs:
        fname = f"{tab.name}.dat"
        print(f"Saving {tab.name} in {fname}. {tab.vector[:10]}")
        data = np.block([t[:, np.newaxis], tab.vector[:, np.newaxis]])
        np.savetxt(fname, X=data, header="time Gk", delimiter=" ")
    if do_plot:
        fig, axes = plt.subplots(nrows=2, sharex="all")
        axes[0].plot(t, vm.vector, label="Vm")
        axes[1].plot(t, inj.vector, label="Im")
        axes[0].legend()
        axes[1].legend()
        if catab is not None:
            fig, ax = plt.subplots()
            fig.suptitle("CaConc")
            ax.plot(t, catab.vector)
        plt.show()


#
# test_granule98.py ends here

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

# import unittest
import logging


LOGLEVEL = os.environ.get("LOGLEVEL", "INFO")
logging.basicConfig(level=LOGLEVEL)


import moose
from moose.neuroml2.reader import NML2Reader


channels = ["NaF", "KDR", "KA", "KC"]


def dump_gate_tables(channel):
    for letter in ['X', 'Y', 'Z']:
        power = getattr(channel, f'{letter}power')
        if power > 0:
            gate = moose.element(f"{channel.path}/gate{letter}")
            minf = gate.tableA / gate.tableB
            mtau = 1 / gate.tableB
            v = np.linspace(gate.min, gate.max, len(minf))
            inf_f = f"{channel.name}.{letter}.inf.dat"
            tau_f = f"{channel.name}.{letter}.tau.dat"
            np.savetxt(
                inf_f, np.block([v[:, np.newaxis], minf[:, np.newaxis]])
            )
            np.savetxt(
                tau_f, np.block([v[:, np.newaxis], mtau[:, np.newaxis]])
            )
            logging.info(f'Saved {letter} gate tables for {channel.name} in {inf_f} and {tau_f}')


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
    nogui = False
    reader = NML2Reader()
    filename = os.path.join(modeldir, "GranuleCell.net.nml")
    reader.read(filename)
    pop_id = reader.doc.networks[0].populations[0].id
    soma = reader.getComp(pop_id, cellIndex=0, segId=0)
    data = moose.Neutral("/data")
    pg = reader.getInput("Gran_10pA")
    inj = moose.Table(f"{data.path}/pulse")
    moose.connect(inj, "requestOut", pg, "getOutputValue")
    vm = moose.Table(f"{data.path}/Vm")
    moose.connect(vm, "requestOut", soma, "getVm")

    for ch in soma.children:
        if moose.isinstance_(ch, moose.HHChannel):
            dump_gate_tables(ch)
                
    simtime = 700e-3
    moose.reinit()
    moose.start(simtime)

    t = np.arange(len(vm.vector)) * vm.dt
    print("%" * 10, len(vm.vector), len(inj.vector))
    results = np.block(
        [t[:, np.newaxis], vm.vector[:, np.newaxis], inj.vector[:, np.newaxis]]
    )
    fname = "Granule_98.dat"
    np.savetxt(fname, X=results, header="time Vm Im", delimiter=" ")
    print(f"Saved results in {fname}")
    if not nogui:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(nrows=2, sharex="all")
        axes[0].plot(t, vm.vector, label="Vm")
        axes[1].plot(t, inj.vector, label="Im")
        plt.legend()
        plt.show()


#
# test_granule98.py ends here

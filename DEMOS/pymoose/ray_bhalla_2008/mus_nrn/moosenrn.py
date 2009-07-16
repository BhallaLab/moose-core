#!/usr/bin/env python

# Author: Subhasis Ray
# 2008-09-05 10:02:31 UTC
import sys
sys.path.append("/home/subha/lib/python2.5/site-packages")
sys.path.append("/home/subha/lib/python2.5/site-packages/neuron")

import time

import pylab
import numpy

import neuron
import moose

#time_file = open("runtime.txt", "w")

class NeuronSim:
    """Wrapper class for the neuron simulation"""
    def __init__(self, fileName="ca3_db.hoc"):
        """Load the file specified by fileName"""
        self.hoc = neuron.h
        self.hoc.load_file(fileName)
        self.hoc.setup()
        
    def run(self, interval):
        """Simulate for interval time in second"""
#	start = time.time()
        self.hoc.run(interval * 1e3) # neuron keeps time in milli second
#	end = time.time()
#	time_file.write("NEURON: " + str(end - start) + "\n")
    
    def cai(self):
        """Returns cai of in nM"""
        return self.hoc.soma(0.5).cai
    
    def cai_record(self):
        """Returns a tuple containing the array of time points and the array
of cai values at the corresponding points"""
        timeVec = numpy.array(neuron.h.vecT)
        caiVec = numpy.array(neuron.h.vecCai)
        return (timeVec, caiVec)
    
    def v_record(self):
        """Returns a tuple containing the array of time points and the array
of membrane potential values at the corresponding points"""
        timeVec = numpy.array(neuron.h.vecT)
        vmVec = numpy.array(neuron.h.vecV)
        return (timeVec, vmVec)

    def update_kconductance(self, factor):
        """Modify the k hcannel conductances in inverse proportion of mapk_star_conc"""
        self.hoc.update_gkbar(factor)
        self.hoc.fcurrent()

    def saveplots(self, suffix):
        cai = "nrn_cai_" + str(suffix) + ".plot"
        vm = "nrn_vm_" + str(suffix) + ".plot"
        t_series, vm_series, = self.v_record()
        t_series, cai_series, = self.cai_record()
        numpy.savetxt(cai, cai_series)
        numpy.savetxt(vm, vm_series)
        numpy.savetxt("nrn_t_" + str(suffix) + ".plot", t_series)

class MooseSim:
    """Wrapper class for moose simulation"""
    volume_scale = 6e20 * 1.257e-16
    def __init__(self, fileName="acc79.g"):
        self._settle_time = 1800.0
        self._ctx = moose.PyMooseBase.getContext()
        self._t_table = []
        self._t = 0.0
        self._ctx.loadG(fileName)
        self.ca_input = moose.Molecule("/kinetics/Ca_input")
        self.mapk_star = moose.Molecule("/kinetics/MAPK*")
        self.pkc_active = moose.Molecule("/kinetics/PKC-active")
        self.pkc_active_table = moose.Table("/graphs/conc2/PKC-active.Co")
        self.pkc_ca_table = moose.Table("/graphs/conc1/PKC-Ca.Co")
        self.mapk_star_table = moose.Table("/moregraphs/conc3/MAPK*.Co")
        self.mapk_star_table.stepMode = 3
        self.mapk_star_table.connect("inputRequest", self.mapk_star, "conc")
        self.mapk_star_table.useClock(2)
        self.ca_input_table = moose.Table("/moregraphs/conc4/Ca_input.Co")
        self.ca_input_table.stepMode = 3
        self.ca_input_table.connect("inputRequest", self.ca_input, "conc")
        self.ca_input_table.useClock(2)
        self._ctx.reset()
        self._ctx.reset()

    def set_ca_input(self, ca):
        """Sets the conc. of Ca_input molecule"""
        print "set_ca_input: BEFORE: nInit =", self.ca_input.nInit, ", n =", self.ca_input.n, ", setting to: ", ca* MooseSim.volume_scale
        self.ca_input.nInit = ca * MooseSim.volume_scale
        self.ca_input.n = ca * MooseSim.volume_scale
        print "set_ca_input: AFTER: nInit =", self.ca_input.nInit, ", n =", self.ca_input.n

    def ca_input(self):
        """Returns scaled value of Ca_input conc."""
        return self.ca_input.conc

    def run(self, interval):
        """Run the simulation for interval time."""
#	start = time.time()
        self._ctx.step(float(interval))
#	end = time.time()
#	time_file.write("MOOSE: " + str(end - start) + "\n")
        # Now expand the list of time points to be plotted
        points = len(self.pkc_ca_table) - len(self._t_table)
        delta = interval * 1.0 / points
        for ii in range(points):
            self._t_table.append(self._t)
            self._t += delta

    def pkc_ca_record(self):
        """Returns the time series for pkc_ca conc."""
        return (self._t_table, self.pkc_ca_table)

    def pkc_active_record(self):
        """Returns time series for pkc_active conc."""
        return (self._t_table, self.pkc_active_table)

    def mapk_star_conc(self):
        """Returns MAPK* conc. in uM"""
        return self.mapk_star.n / MooseSim.volume_scale

    def mapk_star_record(self):
        """Returns time series for [MAPK*]"""
        return (self._t_table, self.mapk_star_table)

    def saveplots(self, suffix):
        pkc_a = "mus_pkc_act_" + str(suffix) + ".plot"
        pkc_ca = "mus_pkc_ca_" + str(suffix) + ".plot"
        mapk_star = "mus_mapk_star_" + str(suffix) + ".plot"
        ca_input = "mus_ca_input_" + str(suffix) + ".plot"
        numpy.savetxt("mus_t_" + str(suffix) + ".plot", self._t_table)
        self.mapk_star_table.dumpFile(mapk_star)
        self.pkc_ca_table.dumpFile(pkc_ca)
        self.pkc_active_table.dumpFile(pkc_a)
        self.ca_input_table.dumpFile(ca_input)
#         numpy.savetxt(pkc_a, numpy.array(self.pkc_active_table))
#         numpy.savetxt(pkc_ca, numpy.array(self.pkc_ca_table))
#         numpy.savetxt(mapk_star,  numpy.array(self.mapk_star_table))
#         numpy.savetxt(ca_input,  numpy.array(self.mapk_star_table))

    def test_run(self):
        self.run(500)
        print "After 500 steps of uninited run: [MAPK*] =", self.mapk_star_conc()
        self.ca_input.nInit = 10 * MooseSim.volume_scale
        self.ca_input.n = 10 * MooseSim.volume_scale
        self.run(5)
        print "After another 5 s with 10uM ca input: [MAPK*] =", self.mapk_star_conc()
        self.ca_input.nInit = 0.08 * MooseSim.volume_scale
        self.ca_input.n = 0.08 * MooseSim.volume_scale
        self.run(500)
        print "finished run. going to plot" 
        print "After another 500 s with 0.08 uM ca input: [MAPK*] =", self.mapk_star_conc()
        pylab.plot(pylab.array(self._t_table),
                   pylab.array(self.pkc_active_table),
                   pylab.array(self._t_table),
                   pylab.array(self.pkc_ca_table))
	pylab.show()
    
def scale_nrncai(cai):
    return (cai - 50e-6) * 4000.0 + 0.08

def saveplot(filename, xx, yy):
    """Save the xx and yy as two columns in a file"""
    outfile = open(filename, 'w')
    for ii in range(len(xx)):
        outfile.write("%13.12G %13.12G\n" % (xx[ii], yy[ii]))
    outfile.close()

if __name__ == "__main__":
#     /* The schedule of experiment is as follows:
#                          ________________                    
#                    1nA  |                | 
#              __         |                |                         __
#      0.15nA |  |        |                |                 0.15nA |  |
#     ________|  |________|                |________________________|  |_   
    
#      1s      0.25s  1s          7s                180s            0.25s 0.05s          
     
#      The 1800 s runs with 1 s intervals interspersed with 1 s of
#      kinetic simulation and update of gkbar for all ca dependent k
#      channels.
#      The genesis model needs over 1 uM [Ca2+] for 10 s.
#     */

    start = time.time()
    mus = MooseSim()
    end = time.time()
#    time_file.write("MOOSE SETUP: " + str(end - start) + "\n")
    mus.set_ca_input(0.08)
    mus.run(1800.0)
    mus.saveplots("1")
    start_mapk = mus.mapk_star_conc()
    start = time.time()
    nrn = NeuronSim()
    end = time.time()
#    time_file.write("NEURON SETUP: " + str(end - start) + "\n")
    nrn.run(2.25)
    nrn.saveplots("1")
    file_ = open("cai.plot", "w")
    
    # Interleaved execution of MOOSE and NEURON model
    # Synchronizing after every 1 s of simulation
    while nrn.hoc.t < 192.25e3:
        scaled_cai = scale_nrncai(nrn.cai())
        mus.set_ca_input(scaled_cai)
        print "scaled_cai =",scaled_cai
        file_.write(str(nrn.hoc.t) + " " + str(nrn.cai()) + " " + str(scaled_cai) +"\n")
        mus.run(1.0)
        gkbar_scale = start_mapk / mus.mapk_star_conc()
        start_mapk = mus.mapk_star_conc()
        print "[mapk*] = ", start_mapk
        nrn.update_kconductance(gkbar_scale)
        nrn.run(1.0)
        print "time is ", nrn.hoc.t , "s"

    file_.close()
    nrn.saveplots("2")
    mus.saveplots("2")
    # final test pulse run
    nrn.run(0.3)
    nrn.saveplots("3")
    t_series, vm_series, = nrn.v_record()
    t_series, cai_series, = nrn.cai_record()
    pylab.subplot(121)
    pylab.plot(t_series, numpy.array(vm_series), t_series, numpy.array(cai_series) * 1e6)
    t_series, pkc_act, = mus.pkc_active_record()
    t_series, pkc_ca, = mus.pkc_ca_record()
    t_series, mapk_star, = mus.mapk_star_record()
    pylab.subplot(122)
    pylab.plot(numpy.array(t_series), numpy.array(pkc_act), numpy.array(t_series), numpy.array(pkc_ca), numpy.array(t_series), numpy.array(mapk_star))
    pylab.show()
#    time_file.close()
    

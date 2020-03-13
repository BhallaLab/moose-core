# Author: @jkopsick (github) https://github.com/jkopsick
#
# Turned into a test case by Dilawar Singh <dilawars@ncbs.res.in>

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import moose
import moose.utils as mu

print("[INFO ] Using moose from %s" % moose.__file__)
import numpy as np
import datetime

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def plot_gate_params(chan, plotpow, VMIN=-0.1, VMAX=0.05, CAMIN=0, CAMAX=1):
    """Plot the gate parameters like m and h of the channel."""
    if chan.className == 'HHChannel':
        print("[INFO ] Testing moose.HHChannel with CaConc")
        cols = 1
        #n=range(0,2,1)
        if chan.Zpower != 0 and (chan.Xpower != 0 or chan.Ypower != 0
                                 ) and chan.useConcentration == True:
            fig, axes = plt.subplots(3, cols, sharex=False)
            axes[1].set_xlabel('voltage')
            axes[2].set_xlabel('Calcium')
        else:
            fig, axes = plt.subplots(2, cols, sharex=True)
            axes[1].set_xlabel('voltage')
        plt.suptitle(chan.name)

        if chan.Xpower > 0:
            assert False, "Should load into gateZ"
            gate = moose.element(chan.path + '/gateX')
            ma = gate.tableA
            mb = gate.tableB
            varray = np.linspace(gate.min, gate.max, len(ma))
            axes[0].plot(varray, 1e3 / mb, label='Xtau ' + chan.name)
            labelpow = '(Xinf)**{}'.format(chan.Xpower)
            infpow = (ma / mb)**chan.Xpower
            label = 'Xinf'
            inf = ma / mb
            axes[1].plot(varray, inf, label=label)
            axes[1].plot(varray, infpow, label=labelpow)
            axes[1].axis([gate.min, gate.max, 0, 1])

        if chan.Ypower > 0:
            assert False, "Should load into gateZ"
            gate = moose.element(chan.path + '/gateY')
            ha = gate.tableA
            hb = gate.tableB
            varray = np.linspace(gate.min, gate.max, len(ha))
            axes[0].plot(varray, 1e3 / hb, label='Ytau ' + chan.name)
            axes[1].plot(varray, ha / hb, label='Yinf ' + chan.name)
            axes[1].axis([gate.min, gate.max, 0, 1])

        if chan.Zpower != 0:
            gate = moose.element(chan.path + '/gateZ')
            gate.min = CAMIN
            gate.max = CAMAX
            za = gate.tableA
            zb = gate.tableB
            Z = za / zb

            # FIXME: These values may not be correct. Once verified, remove
            # this comment or fix the test.
            assert np.isclose(Z.mean(), 0.9954291788156363)
            assert np.isclose(Z.std(), 0.06818901739648629)

            xarray = np.linspace(gate.min, gate.max, len(za))
            if (chan.Xpower == 0
                    and chan.Ypower == 0) or chan.useConcentration == False:
                axes[0].plot(xarray, 1e3 / zb, label='ztau ' + chan.name)
                axes[1].plot(xarray, za / zb, label='zinf' + chan.name)
                if chan.useConcentration == True:
                    axes[1].set_xlabel('Calcium')
            else:
                axes[2].set_xscale("log")
                axes[2].set_ylabel('ss, tau (s)')
                axes[2].plot(xarray, 1 / zb, label='ztau ' + chan.name)
                axes[2].plot(xarray, za / zb, label='zinf ' + chan.name)
                axes[2].legend(loc='best', fontsize=8)
        axes[0].set_ylabel('tau, ms')
        axes[1].set_ylabel('steady state')
        axes[0].legend(loc='best', fontsize=8)
        axes[1].legend(loc='best', fontsize=8)
    else:  #Must be two-D tab channel
        plt.figure()

        ma = moose.element(chan.path + '/gateX').tableA
        mb = moose.element(chan.path + '/gateX').tableB
        ma = np.array(ma)
        mb = np.array(mb)

        plt.subplot(211)

        plt.title(chan.name + '/gateX top: tau (ms), bottom: ss')
        plt.imshow(1e3 / mb,
                   extent=[CAMIN, CAMAX, VMIN, VMAX],
                   aspect='auto',
                   origin='lower')
        plt.colorbar()

        plt.subplot(212)
        if plotpow:
            inf = (ma / mb)**chan.Xpower
        else:
            inf = ma / mb

        plt.imshow(inf,
                   extent=[CAMIN, CAMAX, VMIN, VMAX],
                   aspect='auto',
                   origin='lower')
        plt.xlabel('Ca [mM]')
        plt.ylabel('Vm [V]')
        plt.colorbar()
        if chan.Ypower > 0:
            ha = moose.element(chan.path + '/gateY').tableA
            hb = moose.element(chan.path + '/gateY').tableB
            ha = np.array(ha)
            hb = np.array(hb)

            plt.figure()
            plt.subplot(211)
            plt.suptitle(chan.name + '/gateY tau')
            plt.imshow(1e3 / hb,
                       extent=[CAMIN, CAMAX, VMIN, VMAX],
                       aspect='auto')

            plt.colorbar()
            plt.subplot(212)
            if plotpow:
                inf = (ha / hb)**chan.Ypower
            else:
                inf = ha / hb
            plt.imshow(inf, extent=[CAMIN, CAMAX, VMIN, VMAX], aspect='auto')
            plt.xlabel('Ca [nM]')
            plt.ylabel('Vm [V]')
            plt.colorbar()
    return


def test_nml2(nogui=True):
    global SCRIPT_DIR
    filename = os.path.join(SCRIPT_DIR, 'nml_files/passiveCell.nml')
    mu.info('Loading: %s' % filename)
    nml = moose.mooseReadNML2(filename)
    if not nml:
        mu.warn("Failed to parse NML2 file")
        return

    assert nml, "Expecting NML2 object"
    msoma = nml.getComp(nml.doc.networks[0].populations[0].id, 0, 0)
    data = moose.Neutral('/data')
    pg = nml.getInput('pulseGen1')

    inj = moose.Table('%s/pulse' % (data.path))
    moose.connect(inj, 'requestOut', pg, 'getOutputValue')

    vm = moose.Table('%s/Vm' % (data.path))
    moose.connect(vm, 'requestOut', msoma, 'getVm')

    simtime = 150e-3
    moose.reinit()
    moose.start(simtime)
    print("Finished simulation!")
    yvec = vm.vector
    injvec = inj.vector * 1e12
    m1, u1 = np.mean(yvec), np.std(yvec)
    m2, u2 = np.mean(injvec), np.std(injvec)
    assert np.isclose(m1, -0.0456943), m1
    assert np.isclose(u1, 0.0121968), u1
    assert np.isclose(m2, 26.64890), m2
    assert np.isclose(u2, 37.70607574), u2


def test_nml2_jkopsick():
    # Read the NML model into MOOSE
    filename = os.path.join(SCRIPT_DIR, 'nml_files/MScellupdated_primDend.nml')
    moose.mooseReadNML2(filename, verbose=1)

    # Define the variables needed to view the underlying curves for the channel kinetics
    plot_powers = True
    VMIN = -120e-3
    VMAX = 50e-3
    CAMIN = 0.01e-3
    CAMAX = 40e-3

    # Graph the channel kinetics for the SKCa channel -- the plot doesn't
    # currently show up due to the error
    # in copying the CaPool mechanism, but works if you run it manually after
    # the code completes.
    libchan = moose.element('/library' + '/' + 'SKCa')
    plot_gate_params(libchan, plot_powers, VMIN, VMAX, CAMIN, CAMAX)

    stamp = datetime.datetime.now().isoformat()
    plt.suptitle(stamp, fontsize=6)
    plt.tight_layout()
    plt.savefig(__file__ + ".png")

def test_parse_nml_files():
    import glob
    files = glob.glob(os.path.join(SCRIPT_DIR, 'nml_files', '*.nml'))
    print("Total %s files found" % len(files))
    for f in files:
        if moose.exists('/model'): moose.delete('/model')
        if moose.exists('/library'): moose.delete('/library')
        print("\n\n=========================================")
        print("[INFO ] Reading file %s" % f)
        moose.mooseReadNML2(f)

    quit()
    

def main():
    test_parse_nml_files()
    test_nml2()
    test_nml2_jkopsick()


if __name__ == '__main__':
    main()

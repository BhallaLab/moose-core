import moose
print( 'Using moose from %s' % moose.__file__ )
print( 'Using moose version %s' % moose.__version__ )

import rdesigneur as rd

def test1( ):
    rdes = rd.rdesigneur()
    rdes.buildModel()
    moose.showfields( rdes.soma )

def test2( ):
    if moose.exists( '/model' ):
        moose.delete( '/model' )
    rdes = rd.rdesigneur(
        stimList = [['soma', '1', '.', 'inject', '(t>0.1 && t<0.2) * 2e-8']],
        plotList = [['soma', '1', '.', 'Vm', 'Soma membrane potential']],
    )
    rdes.buildModel()
    moose.reinit()
    moose.start( 0.3 )
    #rdes.display()


def main( ):
    test1( )
    test2( )

if __name__ == '__main__':
    main()

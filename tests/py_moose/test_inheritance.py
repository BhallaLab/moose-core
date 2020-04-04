import moose

print("[INFO ] Using moose from %s" % moose.__file__)


def test_1():
    # Recieved on April 3.
    n = moose.Neutral('/r')
    c = moose.CubeMesh('/r/dend')
    p = moose.Pool('/r/dend/Tiam1')
    r = moose.Reac('/r/dend/CaM_GEF3_Reac')

    grp = moose.Neutral('/r/dend/Ras_gr')
    par = moose.Pool('/r/dend/Ras_gr/CaM_GEF')
    e = moose.Enz('/r/dend/Ras_gr/CaM_GEF/CaM_GEF_RAC_GDP_GTP_enz')

    assert isinstance(n, moose.Neutral), (n, n.__class__)
    assert isinstance(p, moose.PoolBase), p
    assert isinstance(r, moose.ReacBase), r
    assert isinstance(e, moose.EnzBase), e
    moose.delete('/r')


def test_2():
    # Received on April 04
    n = moose.Neutral('/r')
    c = moose.CubeMesh('/r/dend')
    grp = moose.Neutral('/r/dend/Ras_gr')

    # n directly is Neutral object so the result is true
    assert isinstance(n, moose.Neutral), n.__class__
    assert isinstance(c, moose.CubeMesh), c.__class__
    assert isinstance(grp,moose.Neutral), grp.__class__


    # for the same variable name "n", I get the element of the path and when I query
    n = moose.element('/r')
    g = moose.element('/r/dend/Ras_gr')
    assert isinstance(n, moose.Neutral), n.__class__
    assert isinstance(grp, moose.Neutral), grp.__class__

    n1 = moose.element('/r')
    g1 = moose.element('/r/dend/Ras_gr')
    assert isinstance(n, moose.Neutral), n.__class__
    assert isinstance(grp, moose.Neutral), grp.__class__
    assert isinstance(c, moose.CubeMesh), c.__class__


    # Result is same for both p2 and p3
    # But before the commit which I had mentioned early the result are true for
    # all the cases.

def main():
    test_1()
    test_2()


if __name__ == '__main__':
    main()

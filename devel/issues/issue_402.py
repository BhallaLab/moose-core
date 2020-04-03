import moose._moose as _moose
import moose
# from moose import wildcardFind,element,PoolBase,CplxEnzBase,ReacBase,EnzBase,Annotator,exists,Neutral,ConcChan,le
import moose

n = _moose.Neutral('/r')
c = _moose.CubeMesh('/r/dend')
p = _moose.Pool('/r/dend/Tiam1')
r = _moose.Reac('/r/dend/CaM_GEF3_Reac')

grp = moose.Neutral('/r/dend/Ras_gr')
par = moose.Pool('/r/dend/Ras_gr/CaM_GEF')
e = moose.Enz('/r/dend/Ras_gr/CaM_GEF/CaM_GEF_RAC_GDP_GTP_enz')
print(n.className)
#print moose.le('/r')

print("##########")
print(isinstance(n, moose.Neutral))
print(isinstance(p, moose.PoolBase))
print(isinstance(r, moose.ReacBase))
print(isinstance(e, moose.EnzBase))


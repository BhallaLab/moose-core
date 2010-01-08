from moose import *

SIMDT = 1e-5
PLOTDT = 1e-4
GLDT = 1e-2
RUNTIME = 0.05

context = PyMooseBase.getContext()
container = Neutral("/test")
proto_file_name = "../../../DEMOS/gbar/myelin2.p"
context.readCell(proto_file_name, "/test/axon")

gl0 = GLcell("GLcell", container)
gl0.vizpath = '/test/axon'
gl0.port = '9999'
gl0.host = 'localhost'
gl0.attribute = 'Vm'
gl0.threshold = 1
gl0.sync = 'on'
gl0.vscale = 10
gl0.bgcolor = '050050050'
gl0.highvalue = 0.05
gl0.lowvalue = -0.1

context.setClock(0, SIMDT, 0)
context.setClock(1, PLOTDT, 0)
context.setClock(4, GLDT, 0)
context.useClock(4, "/#[TYPE=GLcell]")

context.reset()
context.step(RUNTIME)

from moose import *

SIMDT = 1e-5
PLOTDT = 1e-4
GLDT = 1e-2
RUNTIME = 0.05

context = PyMooseBase.getContext()
container = Neutral("/test")
proto_file_name = "../../../DEMOS/gbar/myelin2.p"
context.readCell(proto_file_name, "/test/axon")

gl0 = GLview("GLview", container)
gl0.vizpath = '/test/axon/##[CLASS=Compartment]'
gl0.host = 'localhost'
gl0.port = '9999'
gl0.bgcolor = '050050050'
gl0.value1 = 'Vm'
gl0.value1min = -0.1
gl0.value1max = 0.05
gl0.morph_val = 1
gl0.color_val = 1
gl0.sync = 'off'
gl0.grid = 'off'

context.setClock(0, SIMDT, 0)
context.setClock(1, PLOTDT, 0)
context.setClock(4, GLDT, 0)
context.useClock(4, "/#[TYPE=GLview]")

context.reset()
context.step(RUNTIME)

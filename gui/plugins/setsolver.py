import moose

def deleteSolver(modelRoot):
	if moose.wildcardFind(modelRoot+'/##[ISA=ChemCompt]'):
		compt = moose.wildcardFind(modelRoot+'/##[ISA=ChemCompt]')
		if ( moose.exists( compt[0].path+'/stoich' ) ):
			st = moose.element(compt[0].path+'/stoich')
			if moose.exists((st.ksolve).path):
				moose.delete(st.ksolve)
			moose.delete( compt[0].path+'/stoich' )
	for x in moose.wildcardFind( modelRoot+'/data/graph#/#' ):
                x.tick = -1
def addSolver(modelRoot,solver):
	compt = moose.wildcardFind(modelRoot+'/##[ISA=ChemCompt]')
	comptinfo = moose.Annotator(moose.element(compt[0]).path+'/info')
	previousSolver = comptinfo.solver
	currentSolver = previousSolver
	if solver == "Gillespie":
		currentSolver = "gssa"
	elif solver == "Runge Kutta":
		currentSolver = "gsl"
	elif solver == "Exponential Euler":
		currentSolver = "ee"
	if previousSolver != currentSolver:
		# if previousSolver != currentSolver
		comptinfo.solver = currentSolver
		if (moose.exists(compt[0].path+'/stoich')):
			# "A: and stoich exists then delete the stoich add solver"
			deleteSolver(modelRoot)
			setCompartmentSolver(modelRoot,currentSolver)
			return True
		else:
			# " B: stoich doesn't exists then addSolver, this is when object is deleted which delete's the solver "
			#  " and solver is also changed, then add addsolver "
			setCompartmentSolver(modelRoot,currentSolver)
			return True
	else:

		# " solver is same "
		if moose.exists(compt[0].path+'/stoich'):
			# " stoich exist, doing nothing"
			return False
		else:
			# "but stoich doesn't exist,this is when object is deleted which deletes the solver
			# " but solver are not changed, then also call addSolver"
			setCompartmentSolver(modelRoot,currentSolver)
			return True
	return False
def setCompartmentSolver(modelRoot,solver):
	compt = moose.wildcardFind(modelRoot+'/##[ISA=ChemCompt]')
	if ( solver == 'gsl' ) or (solver == 'Runge Kutta'):
		ksolve = moose.Ksolve( compt[0].path+'/ksolve' )
	if ( solver == 'gssa' ) or (solver == 'Gillespie'):
		ksolve = moose.Gsolve( compt[0].path+'/gsolve' )
	if ( solver != 'ee' ):
		stoich = moose.Stoich( compt[0].path+'/stoich' )
		stoich.compartment = compt[0]
		stoich.ksolve = ksolve
		stoich.path = compt[0].path+"/##"
	for x in moose.wildcardFind( modelRoot+'/data/graph#/#' ):
		x.tick = 18
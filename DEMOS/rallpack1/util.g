// moose
// genesis

function make_compartment( path, RA, RM, CM, EM, inject, diameter, length, symmetric )
	float RA, RM, CM, EM, inject, diameter, length
	str path
	int symmetric
	
	float PI = 3.141592654
	float Ra = 4.0 * {length} * {RA} / ({PI} * {diameter} * {diameter})
	float Rm = {RM} / ({PI} * {diameter} * {length})
	float Cm = {CM} * ({PI} * {diameter} * {length})
	
	if ( symmetric && GENESIS )
		create symcompartment {path}
	else
		create compartment {path}
	end
	
	setfield {path} \
		Ra {Ra} \
		Rm {Rm} \
		Cm {Cm} \
		Em {EM} \
		inject {inject} \
		dia {diameter} \
		len {length} \
		initVm {EM}
end

function link_compartment( path1, path2, symmetric )
	str path1, path2
	int symmetric
	
	if ( GENESIS )
		if ( symmetric )
			addmsg {path1} {path2} AXIAL Ra previous_state
			addmsg {path2} {path1} RAXIAL Ra previous_state
		else
			addmsg {path1} {path2} AXIAL previous_state
			addmsg {path2} {path1} RAXIAL Ra previous_state
		end
	else
		addmsg {path1}/axial {path2}/raxial
	end
end

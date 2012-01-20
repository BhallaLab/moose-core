cube = bpy.ops.mesh.primitive_cylinder_add()


cube = bpy.context.active_object
cube = bpy.context.active_object.name = 'cylinder'
material = bpy.data.materials.new('blah')

material.diffuse_color = (1.0,0.0,0.0)
cube.active_material = material


cube.active_material.diffuse_color = (0,0,1)
cube.active_material.keyframe_insert(data_path='diffuse_color', frame = 10)

bpy.context.scene.frame_end = 250 


import bpy; import bgl; from mathutils import *; from math import *
material = bpy.data.materials.new('red')
material.diffuse_color = (1.0,0.0,0.0)

def draw_cylinder(name,material,radius,depth,location,rotation,offsetPosition,offsetAngle):

    #bgl.glRotatef(*offsetAngle[:4]) 
    #bgl.glTranslatef(*offsetPosition[:3])

    bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=depth, location=location, rotation=rotation)

    Cylinder = bpy.context.active_object
    Cylinder.name = name
    Cylinder.active_material = material
    Cylinder.rotation_mode = 'AXIS_ANGLE'
    Cylinder.rotation_axis_angle = offsetAngle
    Cylinder.location = offsetPosition


    bgl.glTranslatef(*[i*-1 for i in offsetPosition[:3]])
    bgl.glRotatef(*[i*-1 for i in offsetAngle[:4]])

    return Cylinder

cmpt = draw_cylinder('first',material,radius=1,depth=2,location=(-1,0,0),rotation=(0,0,0),offsetPosition=(10,2,7),offsetAngle=(0.0,0.0,1.0,0.0))


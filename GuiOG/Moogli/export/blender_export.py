import bpy
import bgl
import os
import numpy

def readData():

def setAnimationDefaults():

def drawObjects(morph):      #pass a list of [uniquename,x0,y0,z0,x,y,z,d]/ 's
    material = addMaterial(color = (1.0,0.0,0.0)) #default red initial color
    for cmptmt in morph:
        if cmptmt[1]!=cmptmt[4] or cmptmt[2]!=cmptmt[5] or cmptmt[3]!=cmptmt[6]:

            height = sqrt(pow((cmptmt[4]-cmptmt[1]),2)+pow((cmptmt[5]-cmptmt[2]),2)+pow((cmptmt[6]-cmptmt[3]),2))

            rotation = tuple((asin((cmptmt[4]-cmptmt[1])/sqrt(pow((cmptmt[4]-cmptmt[1]),2)+pow((cmptmt[5]-cmptmt[2]),2))),asin((cmptmt[5]-cmptmt[2])/sqrt(pow((cmptmt[4]-cmptmt[1]),2)+pow((cmptmt[5]-cmptmt[2]),2))),asin((cmptmt[6]-cmptmt[3])/height)))

            startPt = tuple(cmptmt[1:4])
            if len(cmptmt)==8:
                cmpt = drawCylinder(name=cmptmt[0],material,place=startPt,angle=rotation,length=height,radius=cmptmt[7]/2)
            else:
                cmpt = drawCylinder(name=cmptmt[0],material,place=startPt,angle=rotation,length=height,radius=cmptmt[7]/2,cellPosition=tuple(cmptmt[8:11]),cellAngle=tuple(cmptmt[11:]))
        else:
            if len(cmptmt)==8:
                cmpt = drawSphere(name=cmptmt[0],material,place=tuple(cmptmt[1:4]),radius=cmptmt[7]/2)
            else:
                cmpt = drawSphere(name=cmptmt[0],material,place=tuple(cmptmt[1:4]),radius=cmptmt[7]/2,cellPosition=tuple(cmptmt[8:11]))

        sceneObjects.append(cmpt)
    return sceneObjects

def drawSphere(name,material,place,radius,cellPosition=[0.0,0.0,0.0]):
    bgl.glTranslatef(*cellPosition[:3])

    compartment = bpy.ops.mesh.primitive_sphere_add(location = place,radius = radius)
    compartment.name = name
    compartment.active_material = material

    bgl.glTranslatef(*[i*-1 for i in cellPosition[:3]])

    return compartment

def drawCylinder(name,material,place,angle,length,radius,cellPosition=[0.0,0.0,0.0],cellAngle=[0.0,0.0,0.0,0.0]):

    bgl.glRotatef(*cellAngle[:4])
    bgl.glTranslatef(*cellPosition[:3])

    compartment = bpy.ops.mesh.primitive_cylinder_add(radius = radius, depth = length, location = place, rotation = angle)
    compartment.name = name
    compartment.active_material = material

    bgl.glTranslatef(*[i*-1 for i in cellPosition[:3]])
    bgl.glRotatef(*[i*-1 for i in cellAngle[:4]])

    return compartment

def addMaterial(color = (1.0,0.0,0.0)):
    material = bpy.data.materials.new('skin')
    material.diffuse_color = color
    return material

def animateKeyFrames():



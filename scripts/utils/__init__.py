import bpy
import bmesh
from math import sin, cos, pi
tau = 2*pi
import colorsys
import os
import numpy as np

def listify_matrix(matrix):
    matrix_list = []
    for row in matrix:
        matrix_list.append(list(row))
    return matrix_list

def removeObject(obj):
    if obj.type == 'MESH':
        if obj.data.name in bpy.data.meshes:
            bpy.data.meshes.remove(obj.data)
        if obj.name in bpy.context.scene.objects:
            bpy.context.scene.objects.unlink(obj)
        bpy.data.objects.remove(obj)
    else:
        raise NotImplementedError('Other types not implemented yet besides \'MESH\'')


def trackToConstraint(obj, target):
    constraint = obj.constraints.new('TRACK_TO')
    constraint.target = target
    constraint.track_axis = 'TRACK_NEGATIVE_Z'
    #constraint.track_axis = 'TRACK_Z'
    constraint.up_axis = 'UP_Y'
    #constraint.owner_space = 'LOCAL'
    #constraint.target_space = 'LOCAL'

    return constraint


def target(origin=(0,0,0)):
    tar = bpy.data.objects.new('Target', None)
    bpy.context.collection.objects.link(tar)
    tar.location = origin

    return tar


def camera(origin, target=None, fov=60, clip_start=0.1, clip_end=10000, type='PERSP', ortho_scale=6):
    # Create object and camera
    camera = bpy.data.cameras.new("Camera")
    camera.angle = np.deg2rad(fov)
    camera.clip_start = clip_start
    camera.clip_end = clip_end
    camera.type = type # 'PERSP', 'ORTHO', 'PANO'
    if type == 'ORTHO':
        camera.ortho_scale = ortho_scale

    # Link object to scene
    obj = bpy.data.objects.new("CameraObj", camera)
    obj.location = origin
    bpy.context.collection.objects.link(obj)
    bpy.context.scene.camera = obj # Make this the current camera

    if target: trackToConstraint(obj, target)
    return obj


def lamp(origin, type='POINT', energy=1, color=(1,1,1), target=None):
    # Lamp types: 'POINT', 'SUN', 'SPOT', 'HEMI', 'AREA'
    print('createLamp called')
    bpy.ops.object.add(type='LAMP', location=origin)
    obj = bpy.context.object
    obj.data.type = type
    obj.data.energy = energy
    obj.data.color = color

    if target: trackToConstraint(obj, target)
    return obj


def simpleScene(targetCoord, cameraCoord, sunCoord, lens=35):
    print('createSimpleScene called')

    tar = target(targetCoord)
    cam = camera(cameraCoord, tar, lens)
    sun = lamp(sunCoord, 'SUN', target=tar)

    return tar, cam, sun


def setAmbientOcclusion(ambient_occulusion=True, samples=5, blend_type='ADD'):
    # blend_type options: 'ADD', 'MULTIPLY'
    bpy.context.scene.world.light_settings.use_ambient_occlusion = ambient_occulusion
    bpy.context.scene.world.light_settings.ao_blend_type = blend_type
    bpy.context.scene.world.light_settings.samples = samples


def setSmooth(obj, level=None, smooth=True):
    if level:
        # Add subsurf modifier
        modifier = obj.modifiers.new('Subsurf', 'SUBSURF')
        modifier.levels = level
        modifier.render_levels = level

    # Smooth surface
    mesh = obj.data
    for p in mesh.polygons:
        p.use_smooth = smooth

def set_cycles_renderer(scene: bpy.types.Scene,
                        camera_object: bpy.types.Object,
                        num_samples: int,
                        use_denoising: bool = True,
                        use_motion_blur: bool = False,
                        use_transparent_bg: bool = True,
                        gpu_id = 0) -> None:
    scene.camera = camera_object

    #  scene.render.image_settings.file_format = 'PNG'
    scene.render.engine = 'CYCLES'
    scene.render.use_motion_blur = use_motion_blur
    scene.cycles.device = 'GPU'
    scene.render.tile_x = 256
    scene.render.tile_y = 256

    scene.render.film_transparent = use_transparent_bg
    scene.view_layers[0].cycles.use_denoising = use_denoising

    scene.cycles.samples = num_samples

    # GPU config
    cycles_prefs = bpy.context.preferences.addons['cycles'].preferences
    cycles_prefs.compute_device_type = 'CUDA'
    cycles_prefs.get_devices()
    for di, device in enumerate(cycles_prefs.devices):
        device.use = (di == gpu_id)

def rainbowLights(r=5, n=100, freq=2, energy=0.5):
    for i in range(n):
        t = float(i)/float(n)
        pos = (r*sin(tau*t), r*cos(tau*t), r*sin(freq*tau*t))

        # Create lamp
        # bpy.ops.object.add(type='LIGHT', location=pos)
        # obj = bpy.context.object
        # obj.data.type = 'POINT'
        
        # Apply gamma correction for Blender
        color = tuple(pow(c, 2.2) for c in colorsys.hsv_to_rgb(t, 0.6, 1))

        # Set HSV color and lamp energy
        # obj.data.color = color
        # obj.data.energy = energy
        light_data = bpy.data.lights.new(name="light_2.80", type='POINT')
        light_data.color = color
        light_data.energy = energy
        print(energy)
        light_object = bpy.data.objects.new(name="light_2.80", object_data=light_data)
        bpy.context.collection.objects.link(light_object)
        bpy.context.view_layer.objects.active = light_object
        light_object.location = (pos[0], pos[1], pos[2])
    
    # update scene, if needed
    dg = bpy.context.evaluated_depsgraph_get() 
    dg.update()


def removeAll(type=None):
    # Possible type: ‘MESH’, ‘CURVE’, ‘SURFACE’, ‘META’, ‘FONT’, ‘ARMATURE’, ‘LATTICE’, ‘EMPTY’, ‘CAMERA’, ‘LAMP’
    if type:
        bpy.ops.object.select_all(action='DESELECT')
        bpy.ops.object.select_by_type(type=type)
        bpy.ops.object.delete()
    else:
        # Remove all elements in scene
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete(use_global=False)


def simpleMaterial(diffuse_color):
    mat = bpy.data.materials.new('Material')

    # Diffuse
    mat.diffuse_shader = 'LAMBERT'
    mat.diffuse_intensity = 0.9
    mat.diffuse_color = diffuse_color

    # Specular
    mat.specular_intensity = 0

    return mat


def falloffMaterial(diffuse_color):
    mat = bpy.data.materials.new('FalloffMaterial')

    # Diffuse
    mat.diffuse_shader = 'LAMBERT'
    mat.use_diffuse_ramp = True
    mat.diffuse_ramp_input = 'NORMAL'
    mat.diffuse_ramp_blend = 'ADD'
    mat.diffuse_ramp.elements[0].color = (1, 1, 1, 1)
    mat.diffuse_ramp.elements[1].color = (1, 1, 1, 0)
    mat.diffuse_color = diffuse_color
    mat.diffuse_intensity = 1.0

    # Specular
    mat.specular_intensity = 0.0

    # Shading
    mat.emit = 0.05
    mat.translucency = 0.2

    return mat


def colorRGB_256(color):
    return tuple(pow(float(c)/255.0, 2.2) for c in color)


def renderToFolder(renderFolder='rendering', renderName='render', resX=800, resY=800, resPercentage=100, animation=False, frame_end=None):
    print('renderToFolder called')
    scn = bpy.context.scene
    scn.render.resolution_x = resX
    scn.render.resolution_y = resY
    scn.render.resolution_percentage = resPercentage
    if frame_end:
        scn.frame_end = frame_end

    print(bpy.context.space_data)

    # Check if script is executed inside Blender
    if bpy.context.space_data is None:
        # Specify folder to save rendering and check if it exists
        render_folder = os.path.join(os.getcwd(), renderFolder)
        if(not os.path.exists(render_folder)):
            os.mkdir(render_folder)

        if animation:
            # Render animation
            scn.render.filepath = os.path.join(
                render_folder,
                renderName)
            bpy.ops.render.render(animation=True)
        else:
            # Render still frame
            scn.render.filepath = os.path.join(
                render_folder,
                renderName + '.png')
            bpy.ops.render.render(write_still=True)


def render(out_dir, filename,
        depth_file_output,
        normal_file_output,
        mask_file_output,
        resX=800, resY=800, resPercentage=100):
    print('render called')
    scn = bpy.context.scene
    scn.render.resolution_x = resX
    scn.render.resolution_y = resY
    scn.render.resolution_percentage = resPercentage


    # Check if script is executed inside Blender
    if bpy.context.space_data is None:
        os.makedirs(out_dir, exist_ok=True)
 
        # Render still frame
        depth_file_output.file_slots[0].path = filename + '_'
        normal_file_output.file_slots[0].path = filename + '_'
        mask_file_output.file_slots[0].path = filename + '_'

        scn.render.filepath = os.path.join(
            out_dir,
            'rgb',
            filename)
        bpy.ops.render.render(write_still=True)

def bmeshToObject(bm, name='Object'):
    mesh = bpy.data.meshes.new(name+'Mesh')
    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.scene.update()

    return obj
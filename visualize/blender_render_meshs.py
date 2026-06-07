import sys, os, glob, math, argparse, colorsys, numpy as np
import bpy
from mathutils import Euler, Vector


# ------- CLI ------------------------------------------------------
argv = sys.argv[sys.argv.index("--") + 1:]
p = argparse.ArgumentParser()
p.add_argument("inputs", nargs="+")                     # folder or files
p.add_argument("--step", type=int)
p.add_argument("--count", type=int)
p.add_argument("--angle", default=0.0, type=float)
p.add_argument("--radius", type=float, default=5.0, help="camera distance from origin")
p.add_argument("--list", type=str)                      # "0,10,20"
args = p.parse_args(argv)

# ------- collect OBJ paths ---------------------------------------
if len(args.inputs) == 1 and os.path.isdir(args.inputs[0]):
    obj_paths = sorted(glob.glob(os.path.join(args.inputs[0], "frame*.obj")))
else:
    obj_paths = [os.path.abspath(f) for f in args.inputs]

if args.step:
    count = len(obj_paths) if args.count is None else args.count
    obj_paths = obj_paths[:count * args.step:args.step]
if args.list:
    wanted = {int(i) for i in args.list.split(",")}
    obj_paths = [p for p in obj_paths
                 if int(os.path.basename(p)[5:8]) in wanted]

print("Will render:", [os.path.basename(p) for p in obj_paths])

# ------- fresh scene ---------------------------------------------
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# ------- camera & light ------------------------------------------
cam = bpy.data.cameras.new("cam")
cam_o = bpy.data.objects.new("cam_obj", cam)
bpy.context.collection.objects.link(cam_o)
bpy.context.scene.camera = cam_o

light = bpy.data.lights.new("key", type='AREA')
light_o = bpy.data.objects.new("key_light", light)
bpy.context.collection.objects.link(light_o)
light_o.location = (3, -3, -3)

# -------- NEW camera position: azimuth args.angle at elevation 15° ---
az   = args.angle * math.pi             # radians
R    = args.radius
phi  = math.radians(-15)       # fixed 15° elevation

cam_o.location = (
    R * math.cos(phi) * math.sin(az),     # X
   -R * math.cos(phi) * math.cos(az),     # Y  (0 rad → +Y)
    R * math.sin(phi)                     # Z  (15° up)
)

# Aim at the origin
direction = Vector((0, 0, 0)) - cam_o.location
cam_o.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

# ------- colour palette (light → dark blue) ----------------------
def blue_ramp(n):
    h = 210 / 360          # blue hue
    v = np.linspace(0.6, 0.2, n)
    return [(*colorsys.hsv_to_rgb(h, 0.6, val), 0.7) for val in v]

blues = blue_ramp(len(obj_paths))

# ------- import, orient, colour ----------------------------------
for idx, path in enumerate(obj_paths):
    bpy.ops.import_scene.obj(filepath=path)
    for obj in bpy.context.selected_objects:
        #obj.rotation_euler = (math.pi, 0, 0)   # swap Y/Z & upright
        # obj.scale.x = -1  # optional: flip L↔R to face camera

        r, g, b, a = blues[idx]
        mat            = bpy.data.materials.new(f"mat_{idx}")
        mat.diffuse_color       = (r, g, b, a)
        mat.blend_method        = 'BLEND'
        mat.use_backface_culling = False
        obj.data.materials.clear()
        obj.data.materials.append(mat)

# ------- fit camera ----------------------------------------------
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
bpy.ops.view3d.camera_to_view_selected()   # works now (camera exists)

# --- background / sky colour -------------------------------------
bpy.context.scene.render.film_transparent = False          # keep alpha OFF

# reuse existing world if any, otherwise create one
world = bpy.context.scene.world or bpy.data.worlds.new("bright_bg")
bpy.context.scene.world = world

# a) old-style solid colour (works for Workbench / Solid preview)
world.color = (1.0, 1.0, 1.0)      # pure white  (R,G,B) 0-1

# b) if the world uses shader nodes (Cycles/Eevee), set the Background node
world.use_nodes = True
bg = world.node_tree.nodes.get("Background")
if bg is None:
    bg = world.node_tree.nodes.new(type="ShaderNodeBackground")
bg.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1)   # RGBA
bg.inputs["Strength"].default_value = 1.0               # brighten if needed

# ------- render ---------------------------------------------------
out_dir = os.path.dirname(obj_paths[0])
bpy.context.scene.render.filepath = os.path.join(out_dir, f"overlay_{args.angle}.png")
bpy.context.scene.render.image_settings.file_format = 'PNG'
bpy.context.scene.render.resolution_x = 1920
bpy.context.scene.render.resolution_y = 1080
#bpy.context.scene.render.film_transparent = True

bpy.ops.render.render(write_still=True)
print("Saved to", bpy.context.scene.render.filepath)

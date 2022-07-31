import os
from math import radians
from typing import Dict, List, Tuple

import bmesh
import bpy
from bmesh.types import BMesh
from bpy.props import BoolProperty, EnumProperty, StringProperty
from bpy.types import Action, FCurve, Material, Object, Operator
from bpy_extras.io_utils import ImportHelper
from mathutils import Euler, Matrix, Vector

from ..kurohyo_lib import *
from ..util import IterativeDict
from .bone_props import setup_armature, setup_camera
from .coordinate_converter import (focus_to_dist_rotation, fov_to_blender,
                                   pos_to_blender, pos_to_blender_scaled,
                                   rot_to_blender)


class ImportElpk(Operator, ImportHelper):
    """Loads an ELPK file into blender"""
    bl_idname = "import_scene.elpk"
    bl_label = "Import ELPK"

    # Models
    skeleton_type: EnumProperty(
        items=[
            ('NONE', 'Do not load skeleton', ""),
            ('AUTO', 'Auto detect skeleton', ""),
            ('KHEM_A', '[KHEM_A] Man A (Auth/Cabaret)', ""),
            ('KHEM_B', '[KHEM_B] Man B (Battle/Cutscenes)', ""),
            ('KHEM_C', '[KHEM_C] Man C (City/LOD)', ""),
            ('KHEW_A', '[KHEW_A] Woman A (Auth/Cabaret)', ""),
            ('KHEW_B', '[KHEW_B] Woman B (Battle/Cutscenes)', ""),
            ('KHEW_C', '[KHEW_C] Woman C (City/LOD)', ""),
            ('KHEN_N', '[KHEN_N] Neko N (Cat)', ""),
        ],
        name="Skeleton type",
        description="Skeleton to be imported from skeleton.bin",
        default=1)

    look_for_textures: BoolProperty(
        name='Look for textures',
        description='When loading a model, search for a file with the same name with \"TEX\" instead of \"MDL\" and load it',
        default=True,
    )

    enable_backface_culling: BoolProperty(
        name='Enable backface culling',
        description='Do not render backfaces for models that don\'t use them',
        default=True,
    )

    use_alpha_clip: BoolProperty(
        name='Use alpha clip',
        description='Enable alpha clip for materials that are not using alpha blend',
        default=True,
    )

    import_extra_skeletons: BoolProperty(
        name='Import extra skeletons',
        description='Import skeletons found inside the file. These skeletons will not be parented to anything',
        default=False,
    )

    def armature_callback(self, context):
        items = []
        ao = context.active_object
        ao_name = ao.name

        if ao and ao.type == 'ARMATURE':
            # Add the selected armature first so that it's the default value
            items.append((ao_name, ao_name, ""))

        for a in [arm for arm in bpy.data.objects if arm.type == 'ARMATURE' and arm.name != ao_name]:
            items.append((a.name, a.name, ""))
        return items

    # Animation
    armature_name: EnumProperty(
        items=armature_callback,
        name='Target armature',
        description='The armature to use as a base for importing the animation. '
                    'This armature should be from a model from the same game as the animation'
    )

    filter_glob: StringProperty(default="*", options={"HIDDEN"})

    def draw(self, context):
        layout = self.layout

        layout.use_property_split = True
        layout.use_property_decorate = True

        layout.label(text='Models')
        layout.prop(self, 'skeleton_type')
        layout.prop(self, 'look_for_textures')
        layout.prop(self, 'enable_backface_culling')
        layout.prop(self, 'use_alpha_clip')
        layout.prop(self, 'import_extra_skeletons')
        layout.separator()
        layout.label(text='Animation')
        layout.prop(self, 'armature_name')

    def execute(self, context):
        import time

        try:
            start_time = time.time()
            importer = ElpkImporter(self, context, self.filepath, self.as_keywords(ignore=("filter_glob",)))

            importer.read()

            elapsed_s = "{:.2f}s".format(time.time() - start_time)
            print("ELPK import finished in " + elapsed_s)

            return {'FINISHED'}
        except Exception as error:
            print("Catching Error")
            self.report({"ERROR"}, str(error))
        return {'CANCELLED'}


class ElpkImporter:
    def __init__(self, operator: Operator, context: bpy.context, filepath: str, import_settings: dict):
        self.operator = operator
        self.context = context
        self.filepath = filepath
        self.collection = None

        self.skeleton_type = import_settings.get('skeleton_type')
        self.look_for_textures = import_settings.get('look_for_textures')
        self.enable_backface_culling = import_settings.get('enable_backface_culling')
        self.use_alpha_clip = import_settings.get('use_alpha_clip')
        self.import_extra_skeletons = import_settings.get('import_extra_skeletons')

        self.armature_name = import_settings.get('armature_name')

    elpk: Elpk
    collection: bpy.types.Collection

    def check_armature(self):
        """Sets the active object to be the armature chosen by the user"""

        if self.armature_name:
            armature = bpy.data.objects.get(self.armature_name)
            if armature:
                self.context.view_layer.objects.active = armature
                return 0

        # check the active object first
        ao = self.context.active_object
        if ao and ao.type == 'ARMATURE' and ao.data.bones[:]:
            return 0

        # if the active object isn't a valid armature, get its collection and check

        if ao:
            collection = ao.users_collection[0]
        else:
            collection = self.context.view_layer.active_layer_collection

        if collection and collection.name != 'Master Collection':
            meshObjects = [o for o in bpy.data.collections[collection.name].objects
                           if o.data in bpy.data.meshes[:] and o.find_armature()]

            armatures = [a.find_armature() for a in meshObjects]
            if meshObjects:
                armature = armatures[0]
                if armature.data.bones[:]:
                    self.context.view_layer.objects.active = armature
                    return 0

        return "No armature found to add animation to"

    def read(self):
        self.elpk = read_elpk(self.filepath)

        # TODO: use this
        # Load all of the KHBases in the ELPK so we can update the skeleton we import
        # khbases = [p.files[KHBase] for p in self.elpk.pages if KHBase in p.files]

        if any([False] + [p.files[KHPose] for p in self.elpk.pages if KHPose in p.files]):
            if (not self.armature_name) or self.check_armature():
                self.armature_name = None
                print('Warning: Cannot import animation with the selected armature.')

        skeleton_bin_dict = None

        def add_skinned_meshes(full_model_name, skeleton_bin_dict, skinned_meshes, vertex_group_indices, existing_armature=None, existing_khbase=None):
            khskel = khbase = None

            if self.skeleton_type != 'NONE':
                if self.skeleton_type == 'AUTO' and full_model_name[1:].startswith('_KH') and len(full_model_name) > 14:
                    skel_name = f'KHE{full_model_name[5]}_{full_model_name[14]}'
                else:
                    skel_name = self.skeleton_type

                if skel_tuple := skeleton_bin_dict.get(skel_name):
                    khskel, khbase = skel_tuple

            if existing_armature:
                armature_obj = existing_armature
                khbase = existing_khbase
            elif khskel and khbase:
                armature_obj = self.make_armature(khskel, khbase, current_model_name)
            else:
                khbase = KHBase()
                khbase.name = ''
                khbase.bones = dict()

                armature = bpy.data.armatures.new('Armature')
                armature_obj = bpy.data.objects.new('Armature', armature)
                self.collection.objects.link(armature_obj)

            bone_hash_names = {
                hash_fnv0(name): name
                for name in khbase.bones.keys()
            }

            for mesh_obj in skinned_meshes:
                mesh_obj.parent = armature_obj

                # Create the vertex groups for all bones (required)
                # vertex_group_indices is ordered, so the order of the vertex groups here is correct
                for bone_hash in vertex_group_indices.keys():
                    if bone_hash in bone_hash_names:
                        mesh_obj.vertex_groups.new(name=bone_hash_names[bone_hash])

                # Apply the armature modifier
                modifier = mesh_obj.modifiers.new(type='ARMATURE', name="Armature")
                modifier.object = armature_obj

        current_model_name = ''
        current_full_model_name = ''
        skinned_meshes: List[Object] = list()
        vertex_group_indices = IterativeDict()

        armature_obj = None
        page_armature = page_khbase = None

        cam_counter = 0

        for page in self.elpk.pages:
            page_armature = None
            page_khbase = None

            # Import skeletons in the file
            if self.import_extra_skeletons:
                if not self.collection:
                    self.collection = self.make_collection()

                khskel_list: List[KHSkel] = page.files.get(KHSkel)
                khbase_list: List[KHBase] = page.files.get(KHBase)

                # Just load skel and base in pairs
                armatures = list()
                for skel, base in zip(khskel_list, khbase_list):
                    armatures.append(self.make_armature(skel, base, base.name))

                page_armature = armatures[0] if armatures else None
                page_khbase = khbase_list[0] if khbase_list else None

            khcame_list = page.files[KHCame]
            for khpose in page.files[KHPose]:
                if KHPoseFlag.CAMERA in khpose.pose_flags:
                    self.make_camera_action(khpose, cam_counter, khcame_list[0] if khcame_list else None)
                    cam_counter += 1
                elif self.armature_name:
                    self.make_action(khpose)

            for khmig in page.files[KHMig]:
                self.make_image(khmig, page.page_hash)

            khimag_list: List[KHImag] = page.files[KHImag]
            khmate_list: List[KHMate] = page.files[KHMate]

            materials_dict = dict()
            for khimag, khmate in zip(khimag_list, khmate_list):
                materials_dict = self.make_materials(khimag, khmate)

            khmode_list: List[KHMode] = page.files[KHMode]

            # Import the models
            for khmode in khmode_list:
                # This is based on Kurohyo 2 model names
                short_name = khmode.root_node.name[:9] if khmode.root_node.name[1:].startswith(
                    '_KH') else khmode.root_node.name

                if not self.collection:
                    # Only make a collection if there are models
                    self.collection = self.make_collection()

                if short_name != current_model_name:
                    if skinned_meshes:
                        if not skeleton_bin_dict:
                            skeleton_bin_dict = self.import_skeleton_bin()

                        # Create an armature and add the meshes to it
                        add_skinned_meshes(current_full_model_name, skeleton_bin_dict, skinned_meshes,
                                           vertex_group_indices, page_armature, page_khbase)

                    skinned_meshes.clear()
                    vertex_group_indices.clear()

                    # Update the current name
                    current_model_name = short_name
                    current_full_model_name = khmode.root_node.name

                skinned_meshes.extend(self.make_objects(khmode, vertex_group_indices, materials_dict))

        if current_model_name and skinned_meshes:
            if not skeleton_bin_dict:
                skeleton_bin_dict = self.import_skeleton_bin()

            # Create an armature for the last group of meshes
            add_skinned_meshes(current_full_model_name, skeleton_bin_dict, skinned_meshes,
                               vertex_group_indices, page_armature, page_khbase)

        if armature_obj:
            # Set the armature as the active object after importing everything
            self.context.view_layer.objects.active = armature_obj
            bpy.ops.object.mode_set(mode='OBJECT')

        if self.look_for_textures and 'MDL' in os.path.basename(self.filepath):
            head, tail = os.path.split(self.filepath)
            tex_path = os.path.join(head, tail.replace('MDL', 'TEX'))

            if os.path.isfile(tex_path):
                tex_importer = ElpkImporter(self.operator, self.context, tex_path, dict())
                tex_importer.read()

    def make_collection(self) -> bpy.types.Collection:
        """
        Build a collection to hold all of the objects and meshes from the GMDScene.
        :param context: The context used by the import process.
        :return: A collection which the importer can add objects and meshes to.
        """

        collection_name = os.path.basename(self.filepath)
        collection = bpy.data.collections.new(collection_name)

        # Link the new collection to the currently active collection.
        self.context.collection.children.link(collection)
        return collection

    def import_skeleton_bin(self) -> Dict[str, Tuple[KHSkel, KHBase]]:
        """Import skeletons from skeleton.bin"""

        prefs = self.context.preferences

        if 'kurohyo_elpk_blender' not in prefs.addons:
            raise Exception()

        addon_prefs = prefs.addons['kurohyo_elpk_blender'].preferences

        skeleton_bin_type = addon_prefs.skeleton_bin_type
        skeleton_bin_path = addon_prefs.skeleton_bin_path

        if skeleton_bin_type == 'KH1':
            page_indices = {
                'KHEM_A': 46,
                'KHEM_B': 47,
                'KHEM_C': 48,
                'KHEW_A': 40,
                'KHEW_B': 39,
                'KHEW_C': 48,    # KHEW_C should be a duplicate of KHEM_C
            }
        elif skeleton_bin_type == 'KH2':
            page_indices = {
                'KHEM_A': 3,
                'KHEM_B': 4,
                'KHEM_C': 5,
                'KHEN_N': 6,
                'KHEW_A': 2,
                'KHEW_B': 1,
                'KHEW_C': 5,    # KHEW_C should be a duplicate of KHEM_C
            }

        skeleton_dict = dict()

        skel_bin = read_elpk(skeleton_bin_path)
        for name, i in page_indices.items():
            skel_page = skel_bin.pages[i]

            khskel: KHSkel = skel_page.files[KHSkel][0]
            khbase: KHBase = skel_page.files[KHBase][0]

            skeleton_dict[name] = (khskel, khbase)

        # TODO: Enable this
        # # Update the base using other bases found in the elpk
        # if khbase:
        #     list(map(lambda base: khbase.update(base), khbases))

        return skeleton_dict

    def make_armature(self, khskel: KHSkel, khbase: KHBase, armature_name: str):
        armature = bpy.data.armatures.new(armature_name)
        armature.display_type = 'STICK'

        armature_obj = bpy.data.objects.new(armature_name, armature)
        armature_obj.show_in_front = True

        self.collection.objects.link(armature_obj)

        self.context.view_layer.objects.active = armature_obj
        bpy.ops.object.mode_set(mode='EDIT')

        def make_bone(skel_bone: KHSkelBone, base_bone: KHBaseBone, parent_name: str):
            # Convert the node values
            pos = pos_to_blender_scaled(base_bone.location)
            rot = rot_to_blender(base_bone.rotation)
            sca = base_bone.scale

            # Set up the transformation matrix
            this_bone_matrix = Matrix.Translation(pos) @ rot.to_matrix().to_4x4() @ Matrix.Diagonal(sca).to_4x4()

            bone = armature.edit_bones.new(base_bone.name)
            bone.use_relative_parent = False
            bone.use_deform = True

            bone.tail = Vector((0, 0.0001, 0))

            bone.matrix = this_bone_matrix
            bone.parent = armature.edit_bones[parent_name] if parent_name else None

            for child in skel_bone.children:
                make_bone(child, khbase.bones[child.name], base_bone.name)

        make_bone(khskel.root_bone, khbase.bones[khskel.root_bone.name], '')

        bpy.ops.object.mode_set(mode='OBJECT')

        return armature_obj

    def make_objects(self, khmode: KHMode, vertex_group_indices: IterativeDict, materials_dict: Dict[str, Material]) -> List[Object]:
        skinned_meshes = list()

        def make_node(model_node: KHModeNode):
            # TODO: Import clones
            if model_node.is_clone:
                pass

            if model_node.has_model:
                merged_mesh = model_node.merge_meshes()

                mesh_name = model_node.name
                overall_mesh = bpy.data.meshes.new(mesh_name)

                # This list will get filled in khmode_mesh_to_bmesh
                custom_normals = list()
                new_bmesh = self.khmode_mesh_to_bmesh(
                    merged_mesh, model_node.is_skinned, vertex_group_indices, custom_normals)

                # Convert the BMesh to a blender Mesh
                new_bmesh.to_mesh(overall_mesh)
                new_bmesh.free()

                if KHModeVertexFlag.HAS_NORMAL in merged_mesh.vertex_flags:
                    # Use the custom normals we made eariler
                    overall_mesh.create_normals_split()
                    overall_mesh.normals_split_custom_set_from_vertices(custom_normals)
                    overall_mesh.auto_smooth_angle = 0
                    overall_mesh.use_auto_smooth = True

                # Add the material to the mesh
                if material := materials_dict.get(model_node.material_name):
                    if KHModeModelFlag.HAS_ALPHA_BLEND in model_node.model_flags:
                        if not 'kh_alpha_linked' in material:
                            # Link the alpha from the image to the shader node
                            bsdf_node = material.node_tree.nodes.get('Principled BSDF')
                            image_node = material.node_tree.nodes.get('Image Texture')

                            if bsdf_node and image_node:
                                material.node_tree.links.new(image_node.outputs['Alpha'], bsdf_node.inputs['Alpha'])
                                material['kh_alpha_linked'] = True

                        material.blend_method = 'BLEND'

                    if KHModeModelFlag.IS_DUALFACE in model_node.model_flags and self.enable_backface_culling:
                        dualface_name = f'{model_node.material_name}_kh_dualface'
                        if not (dualface_material := materials_dict.get(dualface_name)):
                            dualface_material = materials_dict[dualface_name] = material.copy()
                            dualface_material.name = dualface_name
                            dualface_material.use_backface_culling = False

                        material = dualface_material

                    overall_mesh.materials.append(material)  # materials_dict.get(model_node.material_name))

                mesh_obj: bpy.types.Object = bpy.data.objects.new(mesh_name, overall_mesh)

                # Link the mesh object to the collection
                self.collection.objects.link(mesh_obj)

                # Add the skinned mesh to the list so it can be parented to the armature after it's created
                if model_node.is_skinned:
                    skinned_meshes.append(mesh_obj)

            for child in model_node.children:
                make_node(child)

        make_node(khmode.root_node)

        return skinned_meshes

    def khmode_mesh_to_bmesh(self, mesh: KHModeMesh, is_skinned: bool, vertex_group_indices: IterativeDict, custom_normals: list) -> BMesh:
        bm = bmesh.new()

        if is_skinned:
            deform = bm.verts.layers.deform.new("Vertex Weights")

        # Vertices
        def add_vertex(v: KHModeVertex):
            try:
                # All vertices should have location
                vert = bm.verts.new(pos_to_blender_scaled(v.location))
            except:
                return

            # Normals
            if v.normal:
                normal = pos_to_blender(v.normal)
                custom_normals.append(normal)
                vert.normal = normal

            # Weights
            if is_skinned:
                for i, weight in enumerate(v.weights):
                    if weight > 0:
                        # vert[deform][vertex_group_indices[v.bone_hashes[i]]] = weight
                        vert[deform][vertex_group_indices.get_or_next(v.bone_hashes[i])] = weight

        # Adapted from here:
        # https://github.com/theturboturnip/yk_gmd_io/blob/d88c4a8011ac3af0744e79ae0926f85178d51fea/yk_gmd_blender/blender/importer/mesh_importer.py#L93
        # Find unique (position, normal, boneweight) pairs, assign to BMesh vertex indices
        vert_indices = {}
        merged_idx_to_bmesh_idx: Dict[int, int] = {}
        for i, v in enumerate(mesh.vertices):
            vert_info = (
                v.location.xyz.copy().freeze() if v.location else None,
                v.normal.xyz.copy().freeze() if v.normal else None,
                v.weights,
                v.uv,
            )
            if vert_info in vert_indices:
                merged_idx_to_bmesh_idx[i] = vert_indices[vert_info]
            else:
                next_idx = len(bm.verts)
                vert_indices[vert_info] = next_idx
                merged_idx_to_bmesh_idx[i] = next_idx
                add_vertex(v)

        # Set up the indexing table inside the bmesh so lookups work
        bm.verts.ensure_lookup_table()
        bm.verts.index_update()

        # Faces
        for mesh_face in mesh.faces:
            # Skip "degenerate" triangles
            if len(set(mesh_face)) != 3:
                continue

            try:
                face = bm.faces.new(tuple(map(lambda x: bm.verts[merged_idx_to_bmesh_idx[x]], mesh_face)))
                face.smooth = True
            except Exception as e:
                # We might get duplicate faces
                pass

        bmesh_idx_to_merged_idx = {v: k for k, v in merged_idx_to_bmesh_idx.items()}

        # Color
        if len(mesh.vertices) and mesh.vertices[0].color:
            col_layer = bm.loops.layers.color.new("Color")
            for face in bm.faces:
                for loop in face.loops:
                    color = mesh.vertices[bmesh_idx_to_merged_idx[loop.vert.index]].color
                    loop[col_layer] = tuple(map(lambda x: float(x) / 255, color))

        # UVs
        if len(mesh.vertices) and mesh.vertices[0].uv:
            uv_layer = bm.loops.layers.uv.new('UV')
            for face in bm.faces:
                for loop in face.loops:
                    uv = mesh.vertices[bmesh_idx_to_merged_idx[loop.vert.index]].uv
                    loop[uv_layer].uv = (uv[0], 1.0 - uv[1])

        return bm

    def make_image(self, khmig: KHMig, page_hash):
        image_hash = hex(page_hash)[2:].upper()

        if image := bpy.data.images.get(f'{image_hash}.gim'):
            image.scale(khmig.width, khmig.height)
        else:
            image = bpy.data.images.new(f'{image_hash}.gim', khmig.width, khmig.height)

        width = khmig.width * 4
        pixels = list(khmig.pixels)
        for row in range(khmig.height // 2):
            start = row * width
            start_flipped = (khmig.height - 1 - row) * width

            temp = pixels[start: start + width]
            pixels[start: start + width] = pixels[start_flipped: start_flipped + width]
            pixels[start_flipped: start_flipped + width] = temp

        image.pixels = list(map(lambda x: float(x) / 255, pixels))
        image.update()
        image.pack()

        image['kh_gim_loaded'] = True

        if 'kh_gim_full_name' in image:
            image.name = image['kh_gim_full_name']

    def make_materials(self, khimag: KHImag, khmate: KHMate) -> Dict[str, Material]:
        materials = dict()

        for khmaterial in khmate.materials:
            material: Material = bpy.data.materials.new(khmaterial.name)
            materials[khmaterial.name] = material

            if self.enable_backface_culling:
                material.use_backface_culling = True

            found_texture = False
            for group in khmaterial.groups:
                if found_texture:
                    break

                for texture in group.textures:
                    if texture.is_texture:
                        imag_texture = khimag.textures.get(texture.hash)

                        if not imag_texture:
                            # Just skip
                            continue

                        found_texture = True
                        image_name = imag_texture.name

                        # Remove extension from texture name
                        if image_name.endswith('.tga'):
                            image_name = image_name[:-4]

                        image_hash = hex(hash_fnv0(image_name + '.gim'))[2:].upper()

                        material.use_nodes = True
                        bsdf_node = material.node_tree.nodes.get('Principled BSDF')

                        image_node = material.node_tree.nodes.new('ShaderNodeTexImage')
                        image_node.location = (-300, 220)

                        # Look for the texture with the hash first, then with the loaded name
                        for name in (f'{image_hash}.gim', f'{image_name}.tga', image_name):
                            image = image_node.image = bpy.data.images.get(name)

                            if image:
                                if name == f'{image_hash}.gim' and ('kh_gim_loaded' in image and image['kh_gim_loaded']):
                                    # Rename the image for later use
                                    image_node.image.name = f'{image_name}.tga'
                                break

                        if not image_node.image:
                            # If the texture was not found, make a new empty image to pack the texture into later
                            image_node.image = bpy.data.images.new(f'{image_hash}.gim', 0, 0)
                            image_node.image['kh_gim_loaded'] = False
                            image_node.image['kh_gim_full_name'] = f'{image_name}.tga'

                        # Link the node so the textures appear once the image is loaded
                        material.node_tree.links.new(image_node.outputs['Color'], bsdf_node.inputs['Base Color'])

                        if self.use_alpha_clip:
                            material.node_tree.links.new(image_node.outputs['Alpha'], bsdf_node.inputs['Alpha'])
                            material.blend_method = 'CLIP'
                            material['kh_alpha_linked'] = True

                        # Import only a single texture (per material) for now
                        break

        return materials

    def make_fcurves(self, action: Action, group_name: str, curve: List[KHPoseChannel], data_path: str, convert: bool) -> List[FCurve]:
        if convert:
            # Negate X axis
            if curve[0]:
                for kf in curve[0].keyframes:
                    kf.value *= -1.0

            # Swap Y and Z axes
            temp = curve[1]
            curve[1] = curve[2]
            curve[2] = temp

        fcurves = list()
        for i, channel in enumerate(curve):
            if not (channel and channel.keyframes):
                continue

            frames = list(map(lambda x: x.frame, channel.keyframes))
            values = list(map(lambda x: x.value, channel.keyframes))

            fc = action.fcurves.new(data_path=data_path, index=i, action_group=group_name)
            fc.keyframe_points.add(len(frames))
            fc.keyframe_points.foreach_set('co', [x for co in list(
                map(lambda f, v: (f, v), frames, values)) for x in co])

            for kf in fc.keyframe_points:
                kf.interpolation = 'LINEAR'

            fc.update()

            fcurves.append(fc)

        return fcurves

    def make_camera_action(self, khpose: KHPose, index: int, khcame: KHCame) -> Action:
        # We should expect exactly 1 node
        if not khpose.bones:
            return None

        # There should be only 1 camera node
        if len(khpose.bones) != 1:
            print(f'Warning: Unexpected camera pose node count ({len(khpose.bones)})')

        node = khpose.bones[0]

        action = bpy.data.actions.new(f'{khpose.name}_{str(index).rjust(2, "0")} [{os.path.basename(self.filepath)}]')

        group = action.groups.new("Camera")
        group_name = group.name

        if not self.context.scene.camera and not self.collection:
            # Only make a collection if we will create a new camera
            self.collection = self.make_collection()

        setup_camera(self.context, self.collection)

        roll = node.camera_roll
        fov = node.camera_fov
        focus = node.camera_focus
        location = node.location

        for curve in (location, focus):
            for channel in curve:
                if channel:
                    for kf in channel.keyframes:
                        kf.value /= 10.0

        location_curves = self.make_fcurves(action, group_name, location, 'location', True)

        # Import focus and roll so we can use fcurve.evaluate()
        focus_curves = self.make_fcurves(action, group_name, focus, 'temp_focus', True)
        roll_curves = self.make_fcurves(action, group_name, [roll], 'temp_roll', False)
        roll_curve = roll_curves[0] if roll_curves else None

        end_frame = max(0.0, *list(map(lambda x: x.range()[1], location_curves)),
                        *list(map(lambda x: x.range()[1], focus_curves)))

        # Fill the curves list
        for curves_list in (location_curves, focus_curves):
            for x in [i for i in range(3) if i not in [fc.array_index for fc in curves_list]]:
                curves_list.insert(x, None)

        # Create focus distance channel
        dists = KHPoseChannel()
        dists.keyframes = list()

        rotation: List[KHPoseChannel] = list()
        for i in range(4):
            channel = KHPoseChannel()
            channel.keyframes = list()
            rotation.append(channel)

        # Evaluate the quaternions
        for f in range(int(end_frame)):
            loc = Vector(tuple(map(lambda fc: fc.evaluate(float(f)) if fc else 0.0, location_curves)))
            foc = Vector(tuple(map(lambda fc: fc.evaluate(float(f)) if fc else 0.0, focus_curves)))
            rol = roll_curve.evaluate(float(f)) if roll_curve else 0.0

            dist, rot = focus_to_dist_rotation(loc, foc, rol)

            # Add focus distance keyframe
            kf = KHPoseKeyframe()
            kf.frame = f
            kf.value = dist
            dists.keyframes.append(kf)

            for i in range(4):
                kf = KHPoseKeyframe()
                kf.frame = f
                kf.value = rot[i]
                rotation[i].keyframes.append(kf)

        # Remove focus point and roll curves
        for fc in focus_curves:
            action.fcurves.remove(fc)

        if roll_curve:
            action.fcurves.remove(roll_curve)

        self.make_fcurves(action, group_name, [dists], 'data.dof.focus_distance', False)
        rotation_curves = self.make_fcurves(action, group_name, rotation, 'rotation_quaternion', False)

        if fov:
            for kf in fov.keyframes:
                kf.value = fov_to_blender(radians(kf.value))

            self.make_fcurves(action, group_name, [fov], 'data.lens', False)

        if rotation_curves:
            fc = action.fcurves.new(data_path=f'rotation_mode', action_group=group_name)
            fc.keyframe_points.add(1)
            fc.keyframe_points[0].co = (0.0, 0)  # 0 == QUATERNION
            fc.update()

        self.context.scene.frame_start = 0

        return action

    def make_action(self, khpose: KHPose) -> Action:
        action = bpy.data.actions.new(f'{khpose.name} [{os.path.basename(self.filepath)}]')

        ao = self.context.active_object
        bone_props = setup_armature(ao)

        # NOTE: skeletons usually don't have initial rotation, so we can skip
        # multiplying those with the bone props

        # NOTE: Direction bone does not need to be merged, but why is it there?

        if khpose.bones:
            # Rename root bone
            khpose.bones[0].name = list(bone_props.keys())[0]

        for bone in khpose.bones:
            if bone.name not in ao.pose.bones:
                print(f'Warning: skipping non existing bone \"{bone.name}\"')
                continue

            group_name = action.groups.new(bone.name).name

            bone_path = f'pose.bones[\"{bone.name}\"]'

            scale = bone.scale
            rotation_euler = bone.rotation
            location = bone.location

            bone_prop = bone_props.get(bone.name)

            bone_loc = bone_prop.loc if bone_prop else Vector()
            bone_loc = (-bone_loc.x, bone_loc.z, bone_loc.y)

            parent_loc = bone_props[bone_prop.parent_name].loc if bone_prop and bone_prop.parent_name in bone_props else Vector()
            parent_loc = (-parent_loc.x, parent_loc.z, parent_loc.y)

            for i, channel in enumerate(location):
                if channel:
                    for kf in channel.keyframes:
                        kf.value /= 10.0
                        kf.value = kf.value - bone_loc[i] + parent_loc[i]

            for curve, path in ((location, 'location'), (scale, 'scale')):
                self.make_fcurves(action, group_name, curve, f'{bone_path}.{path}', True)

            rotation_curves = self.make_fcurves(action, group_name, rotation_euler, f'{bone_path}.rotation_euler', True)

            # Fill the curves list
            for x in [i for i in range(3) if i not in [fc.array_index for fc in rotation_curves]]:
                rotation_curves.insert(x, None)

            rotation: List[KHPoseChannel] = list()
            for i in range(4):
                channel = KHPoseChannel()
                channel.keyframes = list()
                rotation.append(channel)

            # Evaluate the quaternions
            for f in range(int(khpose.end_frame)):
                rot = Euler(tuple(map(lambda fc: fc.evaluate(float(f))
                                      if fc else 0.0, rotation_curves)), 'XZY').to_quaternion()

                for i in range(4):
                    kf = KHPoseKeyframe()
                    kf.frame = f
                    kf.value = rot[i]
                    rotation[i].keyframes.append(kf)

            self.make_fcurves(action, group_name, rotation, f'{bone_path}.rotation_quaternion', False)

            for fc in rotation_curves:
                if fc:
                    action.fcurves.remove(fc)

        # self.context.scene.render.fps = 30
        self.context.scene.frame_start = 0
        # self.context.scene.frame_current = 0
        # self.context.scene.frame_end = int(end_frame)

        return action


def menu_func_import_kurohyo(self, context):
    self.layout.operator(ImportElpk.bl_idname, text='Kurohyo ELPK Container')

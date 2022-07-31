from typing import Dict, List

import bpy
from bpy.types import EditBone
from mathutils import Quaternion, Vector


class KHBlenderBoneProps:
    loc: Vector
    rot: Quaternion
    parent_name: str

    # Default props
    def __init__(self):
        self.loc = Vector()
        self.rot = Quaternion()
        self.parent_name = ''


def setup_camera(context: bpy.context, collection: bpy.types.Collection):
    camera = context.scene.camera

    if not camera:
        camera_data = bpy.data.cameras.new(name='Camera')
        camera = bpy.data.objects.new('Camera', camera_data)
        collection.objects.link(camera)

    if not camera.animation_data:
        camera.animation_data_create()
    
    return camera


def setup_armature(ao: bpy.types.Object) -> Dict[str, KHBlenderBoneProps]:
    if not ao.animation_data:
        ao.animation_data_create()

    hidden = ao.hide_get()
    mode = ao.mode

    # Necessary steps to ensure proper importing
    ao.hide_set(False)
    bpy.ops.object.mode_set(mode='POSE')
    bpy.ops.pose.select_all(action='SELECT')
    bpy.ops.pose.transforms_clear()
    bpy.ops.pose.select_all(action='DESELECT')

    bone_props = get_edit_bones_props(ao)

    bpy.ops.object.mode_set(mode=mode)
    ao.hide_set(hidden)

    return bone_props


def get_edit_bones_props(ao: bpy.types.Object) -> Dict[str, KHBlenderBoneProps]:
    bpy.ops.object.mode_set(mode='EDIT')

    bone_props = get_bones_props(ao.data.edit_bones)

    bpy.ops.object.mode_set(mode='POSE')
    return bone_props


def get_bones_props(edit_bones: List[EditBone]) -> Dict[str, KHBlenderBoneProps]:
    bone_props = {}

    for b in edit_bones:
        bp = KHBlenderBoneProps()
        bp.parent_name = b.parent.name if b.parent else ""

        bp.loc = b.matrix.to_translation()
        bp.rot = b.matrix.to_quaternion()

        bone_props[b.name] = bp
    return bone_props

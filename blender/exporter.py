from typing import Dict, List
from copy import deepcopy

import bpy
from bpy.props import EnumProperty, StringProperty
from bpy.types import Action, FCurve, Operator
from bpy_extras.io_utils import ExportHelper
from mathutils import Euler, Quaternion, Vector

from ..kurohyo_lib import *
from .bone_props import KHBlenderBoneProps, get_edit_bones_props


class ExportKHPose(Operator, ExportHelper):
    """Exports an animation to the Kurohyou pose format"""
    bl_idname = "export_scene.pose"
    bl_label = "Export Kurohyou pose"

    filter_glob: StringProperty(default="*.pose", options={"HIDDEN"})

    # Don't force a file extension
    filename_ext = '.pose'
    check_extension = None

    def action_callback(self, context: bpy.context):
        items = []

        # TODO: Instead of setting the default action to the one used by the active object,
        # maybe we should use the one used by the selected armature_name?
        action_name = ""
        ao = context.active_object if self.export_format != 'came' else context.scene.camera
        if ao and ao.animation_data:
            # Add the selected action first so that it's the default value
            selected_action = ao.animation_data.action
            if selected_action:
                action_name = selected_action.name
                items.append((action_name, action_name, ""))

        for a in [act for act in bpy.data.actions if act.name != action_name]:
            items.append((a.name, a.name, ""))
        return items

    def armature_callback(self, context: bpy.context):
        items = []

        if self.export_format == 'came':
            ao = context.scene.camera
            obj_type = 'CAMERA'
        else:
            ao = context.active_object
            obj_type = 'ARMATURE'

        ao_name = ao.name if ao else ''

        if ao and ao.type == obj_type:
            # Add the selected armature first so that it's the default value
            items.append((ao_name, ao_name, ""))

        for a in [arm for arm in bpy.data.objects if arm.type == obj_type and arm.name != ao_name]:
            items.append((a.name, a.name, ""))
        return items

    def action_update(self, context: bpy.context):
        name = self.action_name
        if '(' in name and ']' in name:
            # Used to avoid suffixes (e.g ".001")
            pose_name: str = name[:name.index('(')].strip()

            if pose_name.startswith('ex'):
                pose_name = pose_name[2:]

            # Set the file name
            for screenArea in context.window.screen.areas:
                if screenArea.type == 'FILE_BROWSER':
                    params = screenArea.spaces[0].params
                    params.filename = f'{pose_name}.pose'
                    break

    export_format: EnumProperty(
        items=[('pose', 'Model Animation', ''),
               ('came', 'Camera Animation', ''),
               ],
        name="Export Format",
        description="The animation format to export as",
        default=0,
    )

    action_name: EnumProperty(
        items=action_callback,
        name="Action",
        description="The action to be exported",
        default=0,
        update=action_update,
    )

    armature_name: EnumProperty(
        items=armature_callback,
        name="Armature",
        description="The armature which the action will use as a base",
    )

    def draw(self, context):
        layout = self.layout

        layout.use_property_split = True
        layout.use_property_decorate = True  # No animation.

        layout.prop(self, 'export_format')
        layout.separator()
        layout.prop(self, 'armature_name')
        layout.prop(self, 'action_name')
        layout.separator()

        self.action_update(context)

    def execute(self, context):
        import time

        try:
            if self.export_format == 'pose':
                arm = self.check_armature(context)
                if isinstance(arm, str):
                    raise Exception(arm)

            start_time = time.time()
            exporter = KHPoseExporter(context, self.filepath, self.as_keywords(ignore=("filter_glob",)))
            exporter.export()

            elapsed_s = "{:.2f}s".format(time.time() - start_time)
            print("Export finished in " + elapsed_s)

            self.report({"INFO"}, f"Finished exporting {exporter.action_name}")
            return {'FINISHED'}
        except Exception as error:
            print("Catching Error")
            self.report({"ERROR"}, str(error))
        return {'CANCELLED'}

    def check_armature(self, context: bpy.context):
        """Sets the active object to be the armature chosen by the user"""

        if self.armature_name:
            armature = bpy.data.objects.get(self.armature_name)
            if armature:
                context.view_layer.objects.active = armature
                return 0

        # check the active object first
        ao = context.active_object
        if ao and ao.type == 'ARMATURE' and ao.data.bones[:]:
            return 0

        # if the active object isn't a valid armature, get its collection and check

        if ao:
            collection = ao.users_collection[0]
        else:
            collection = context.view_layer.active_layer_collection

        if collection and collection.name != 'Master Collection':
            meshObjects = [o for o in bpy.data.collections[collection.name].objects
                           if o.data in bpy.data.meshes[:] and o.find_armature()]

            armatures = [a.find_armature() for a in meshObjects]
            if meshObjects:
                armature = armatures[0]
                if armature.data.bones[:]:
                    context.view_layer.objects.active = armature
                    return 0

        return "No armature found to get animation from"


class KHPoseExporter:
    def __init__(self, context: bpy.context, filepath, export_settings: Dict):
        self.filepath = filepath
        self.context = context

        self.action_name = export_settings.get("action_name")
        self.is_camera = export_settings.get("export_format") == 'came'

        self.pose = KHPose()
        self.came = KHCame() if self.is_camera else None

    bone_props: Dict[str, KHBlenderBoneProps]

    def clean_action_name(self, name):
        # Example action name: "MB_W0105 (DB522C19) [heat_30200]"
        if '(' in name and ']' in name:
            return name[:name.index('(')].strip()
        return name

    def export(self):
        print(f"Exporting action: {self.action_name}")

        # Active object was set correctly during operator execution
        self.ao = self.context.active_object
        if not self.ao:
            raise Exception('Active object not found')
        elif not self.is_camera and self.ao.type != 'ARMATURE':
            raise Exception('Armature not found')

        self.bone_props = get_edit_bones_props(self.ao) if not self.is_camera else None

        elpk_page: list

        if self.is_camera:
            # Initialize self.pose and self.came
            self.make_camera()
            elpk_page = [self.pose, self.came]
        else:
            # Initialize self.pose
            self.make_anm()
            elpk_page = [self.pose]

        file_bytes = write_elpk_page(elpk_page)

        with open(self.filepath, 'wb') as f:
            f.write(file_bytes)

        print("Animation export finished")

    def make_camera(self):
        pass

    def make_anm(self):
        action: Action = bpy.data.actions.get(self.action_name)

        if not action:
            raise Exception('Action not found')

        self.pose.name = 'ex' + self.clean_action_name(self.action_name)

        self.pose.bones = list()
        for group in action.groups.values():
            # Skip direction bone
            if group.name == 'Direction':
                continue

            self.pose.bones.append(self.make_bone(group.name, group.channels))

        # Rename root bone
        self.pose.bones[0].name = self.pose.name

        # Find hip bone
        hip_bones = [b for b in self.pose.bones if b.name == 'hip']

        # Direction bone
        if len(hip_bones):
            hip_bone = hip_bones[0]
            direction_bone = deepcopy(hip_bone)

            direction_bone.name = 'Direction'

            # Direction bone only has X and Z axes of location
            direction_bone.location[1] = None
            direction_bone.rotation = [None] * 3
            direction_bone.scale = [None] * 3

            direction_bone.initial_location = Vector()
            direction_bone.initial_rotation = Euler()
            direction_bone.initial_scale = Vector((1.0, 1.0, 1.0))

            self.pose.bones.append(direction_bone)

    def make_bone(self, bone_name: str, channels: List[FCurve]) -> KHPoseBone:
        bone = KHPoseBone()
        bone.name = bone_name

        # (w,) x, y, z
        loc_curves: List[FCurve] = [None] * 3
        rot_curves: List[FCurve] = [None] * 4
        sca_curves: List[FCurve] = [None] * 3

        channel_dict = {
            'location': loc_curves,
            'rotation_quaternion': rot_curves,
            'scale': sca_curves,
        }

        for c in channels:
            # Data path without bone name
            data_path = c.data_path[c.data_path.rindex('.') + 1:] if '.' in c.data_path else ''

            if curves := channel_dict.get(data_path):
                curves[c.array_index] = c
            else:
                print(f'Warning: Ignoring curve with unsupported data path {c.data_path} and index {c.array_index}')

        # ------------------- location -------------------
        bone_prop = self.bone_props.get(bone.name)
        bone_loc = bone_prop.loc if bone_prop else Vector()
        parent_loc = (
            self.bone_props[bone_prop.parent_name].loc
            if bone_prop and bone_prop.parent_name in self.bone_props
            else Vector())

        for i in range(3):
            if not loc_curves[i]:
                continue

            # Create channel
            bone.location[i] = KHPoseChannel()

            # Get keyframes
            axis_co = [0] * 2 * len(loc_curves[i].keyframe_points)
            loc_curves[i].keyframe_points.foreach_get('co', axis_co)

            axis_iter = iter(axis_co)
            for frame, value in zip(axis_iter, axis_iter):
                kf = KHPoseKeyframe()
                kf.frame = frame
                kf.value = (value + bone_loc[i] - parent_loc[i]) * 10.0

                # Negate X axis
                if i == 0:
                    kf.value = -kf.value

                bone.location[i].keyframes.append(kf)

        # Swap Y and Z axes
        temp = bone.location[1]
        bone.location[1] = bone.location[2]
        bone.location[2] = temp

        # ------------------- rotation -------------------
        if any(rot_curves):
            for i in range(3):
                bone.rotation[i] = KHPoseChannel()

            # Evaluate the euler angles
            end_frame = int(max([c.keyframe_points[-1].co[0] for c in rot_curves if c and len(c.keyframe_points)]))
            for f in range(end_frame + 1):
                rot = Quaternion(tuple(map(lambda fc: fc.evaluate(float(f))
                                 if fc else 0.0, rot_curves))).to_euler('XZY')

                for i in range(3):
                    kf = KHPoseKeyframe()
                    kf.frame = f
                    kf.value = rot[i] if (i != 0) else -rot[i]    # Negate X axis
                    bone.rotation[i].keyframes.append(kf)

            # Swap Y and Z axes
            temp = bone.rotation[1]
            bone.rotation[1] = bone.rotation[2]
            bone.rotation[2] = temp

        # ------------------- scale -------------------
        for i in range(3):
            if not sca_curves[i]:
                continue

            # Create channel
            bone.scale[i] = KHPoseChannel()

            # Get keyframes
            axis_co = [0] * 2 * len(sca_curves[i].keyframe_points)
            sca_curves[i].keyframe_points.foreach_get('co', axis_co)

            axis_iter = iter(axis_co)
            for frame, value in zip(axis_iter, axis_iter):
                kf = KHPoseKeyframe()
                kf.frame = frame
                kf.value = value

                # Negate X axis
                if i == 0:
                    kf.value = -kf.value

                bone.scale[i].keyframes.append(kf)

        # Swap Y and Z axes
        temp = bone.scale[1]
        bone.scale[1] = bone.scale[2]
        bone.scale[2] = temp

        # Initial location/rotation/scale
        # This should instead be used to store the value of channels that have a single keyframe
        # (i.e. channels that were originally created from initial location/rotation/scale)
        bone.initial_location = Vector([c.keyframes[0].value if c else 0.0 for c in bone.location])
        bone.initial_rotation = Euler([c.keyframes[0].value if c else 0.0 for c in bone.rotation])
        bone.initial_scale = Vector([c.keyframes[0].value if c else 1.0 for c in bone.scale])

        return bone


def menu_func_export_khpose(self, context):
    self.layout.operator(ExportKHPose.bl_idname, text='Kurohyou Animation (.pose)')

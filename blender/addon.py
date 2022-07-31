import bpy
from bpy.props import EnumProperty, StringProperty
from bpy.types import AddonPreferences

from .importer import ImportElpk, menu_func_import_kurohyo


class KHSkeletonPreferences(AddonPreferences):
    bl_idname = 'kurohyo_elpk_blender'

    skeleton_bin_type: EnumProperty(
        name='Game for skeleton.bin',
        description='Game from which the skeleton file was taken',
        items=[
            ('KH1', 'Kurohyo 1', ''),
            ('KH2', 'Kurohyo 2', ''),
        ],
        default=1,
    )

    skeleton_bin_path: StringProperty(
        name='Path to skeleton.bin',
        description='Path to the skeleton file. In Kurohyo 1, this is \"globals.bin\".\nIn Kurohyo 2, this is \"skeleton.bin\"',
        subtype='FILE_PATH',
        )

    def draw(self, context):
        layout = self.layout

        layout.use_property_split = True
        layout.use_property_decorate = True

        layout.label(text='Choose the game type and path to skeleton.bin. Required for importing models.')
        layout.prop(self, 'skeleton_bin_type')
        layout.prop(self, 'skeleton_bin_path')


classes = (
    KHSkeletonPreferences,
    ImportElpk,
)


def register():
    for c in classes:
        bpy.utils.register_class(c)

    # Add to the import menu
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import_kurohyo)


def unregister():
    # Remove from the import menu
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import_kurohyo)

    for c in reversed(classes):
        bpy.utils.unregister_class(c)

# Include the bl_info at the top level always
bl_info = {
    "name": "Kurohyo ELPK File Import",
    "author": "SutandoTsukai181",
    "version": (1, 1, 0),
    "blender": (2, 93, 0),
    "location": "File > Import-Export",
    "description": "Import Kurohyo ELPK Models and Animations",
    "warning": "",
    "doc_url": "https://github.com/SutandoTsukai181/kurohyo_elpk_blender",
    "tracker_url": "https://github.com/SutandoTsukai181/kurohyo_elpk_blender/issues",
    "category": "Import-Export",
}

import bpy
from .blender.addon import register, unregister

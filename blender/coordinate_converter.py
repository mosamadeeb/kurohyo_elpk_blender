from math import tan
from typing import Tuple

from mathutils import Quaternion, Vector, Euler


def pos_to_blender(pos):
    return Vector([-pos[0], pos[2], pos[1]])

def pos_from_blender(pos):
    return pos_to_blender(pos)

def pos_to_blender_scaled(pos):
    return pos_to_blender(list(map(lambda x: x / 10.0, pos)))

def pos_from_blender_scaled(pos):
    return pos_from_blender(list(map(lambda x: x * 10.0, pos)))

def rot_to_blender(rot):
    return Euler(pos_to_blender(rot[:])[:]).to_quaternion()


def fov_to_blender(fov, sensor=20.0):
    # This sensor value looks correct using the default blender camera settings (Auto, Sensor size 36mm)
    return (sensor / 2) / tan(fov / 2)


def focus_to_dist_rotation(location: Vector, focus_point: Vector, roll: float) -> Tuple[float, Quaternion]:
    forward: Vector = focus_point - location
    dist = forward.length

    forward.normalize()

    rotation = forward.to_track_quat('-Z', 'Y')
    rotation = rotation @ Euler((0, 0, roll)).to_quaternion()

    return (dist, rotation)

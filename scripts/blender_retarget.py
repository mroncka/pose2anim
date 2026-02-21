"""Blender retargeting script for pose2anim BVH output.

Usage:
    blender --background --python scripts/blender_retarget.py -- \
        --bvh animation.bvh \
        --character model.fbx \
        --output animated_model.fbx
"""

import sys
import argparse


def main():
    # Parse args after "--"
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="Retarget BVH to 3D character")
    parser.add_argument("--bvh", required=True, help="Input BVH file")
    parser.add_argument("--character", required=True, help="Character FBX/blend file")
    parser.add_argument("--output", required=True, help="Output FBX file")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale factor")
    args = parser.parse_args(argv)

    import bpy

    # Clear scene
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # Import BVH
    bpy.ops.import_anim.bvh(filepath=args.bvh)
    bvh_armature = bpy.context.active_object
    print(f"Imported BVH armature: {bvh_armature.name}")

    # Import character
    ext = args.character.rsplit(".", 1)[-1].lower()
    if ext == "fbx":
        bpy.ops.import_scene.fbx(filepath=args.character)
    elif ext == "blend":
        with bpy.data.libraries.load(args.character) as (data_from, data_to):
            data_to.objects = data_from.objects
        for obj in data_to.objects:
            bpy.context.collection.objects.link(obj)

    # Find character armature
    char_armature = None
    for obj in bpy.context.scene.objects:
        if obj.type == "ARMATURE" and obj != bvh_armature:
            char_armature = obj
            break

    if char_armature is None:
        print("ERROR: No character armature found")
        sys.exit(1)

    print(f"Character armature: {char_armature.name}")

    # Basic retargeting: copy rotation constraints
    bpy.context.view_layer.objects.active = char_armature
    bpy.ops.object.mode_set(mode="POSE")

    for bone in char_armature.pose.bones:
        # Try to find matching bone in BVH armature
        bvh_bone_name = bone.name  # Same name mapping (simplistic)
        if bvh_bone_name in bvh_armature.pose.bones:
            constraint = bone.constraints.new("COPY_ROTATION")
            constraint.target = bvh_armature
            constraint.subtarget = bvh_bone_name
            print(f"  Mapped: {bone.name} ← {bvh_bone_name}")

    bpy.ops.object.mode_set(mode="OBJECT")

    # Bake animation
    bpy.context.view_layer.objects.active = char_armature
    bpy.ops.nla.bake(
        frame_start=bpy.context.scene.frame_start,
        frame_end=bpy.context.scene.frame_end,
        only_selected=False,
        visual_keying=True,
        clear_constraints=True,
        bake_types={"POSE"},
    )

    # Export FBX
    bpy.ops.export_scene.fbx(
        filepath=args.output,
        use_selection=False,
        add_leaf_bones=False,
        bake_anim=True,
    )
    print(f"Exported: {args.output}")


if __name__ == "__main__":
    main()

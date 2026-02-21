"""FBX animation export via Blender Python API.

This module provides FBX export functionality using Blender's
Python API (bpy). It can be run as a Blender script or imported
when Blender's Python is available.

Usage:
    blender --background --python -c "from pose2anim.export.fbx_exporter import export_fbx; export_fbx('anim.bvh', 'output.fbx')"
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def export_fbx(bvh_path: str, output_path: str, character_path: str = None):
    """Convert BVH to FBX, optionally retargeting to a character.

    Args:
        bvh_path: Path to input BVH file.
        output_path: Path for output FBX file.
        character_path: Optional path to character FBX/blend file for retargeting.
    """
    try:
        import bpy
    except ImportError:
        logger.error(
            "Blender Python API (bpy) not available. "
            "Run this script through Blender: "
            "blender --background --python scripts/blender_retarget.py"
        )
        raise

    # Clear scene
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # Import BVH
    bpy.ops.import_anim.bvh(filepath=str(bvh_path))
    logger.info(f"Imported BVH: {bvh_path}")

    if character_path:
        # Import character model
        ext = Path(character_path).suffix.lower()
        if ext == ".fbx":
            bpy.ops.import_scene.fbx(filepath=str(character_path))
        elif ext in (".blend",):
            bpy.ops.wm.append(filepath=str(character_path))

        logger.info(f"Imported character: {character_path}")
        # TODO: Automatic bone mapping and retargeting

    # Export FBX
    bpy.ops.export_scene.fbx(
        filepath=str(output_path),
        use_selection=False,
        add_leaf_bones=False,
        bake_anim=True,
        bake_anim_use_all_bones=True,
    )
    logger.info(f"Exported FBX: {output_path}")

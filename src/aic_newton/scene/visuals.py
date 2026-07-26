"""Load render-only meshes, materials, and lighting."""

from pathlib import Path
from xml.etree import ElementTree as ET

import newton
import numpy as np
import warp as wp

from ..config import VIEWER


def _hide_shape_range(builder: newton.ModelBuilder, start: int) -> None:
    """Hide shapes added after an index without changing collision flags."""
    visible = int(newton.ShapeFlags.VISIBLE)
    for shape in range(start, builder.shape_count):
        builder.shape_flags[shape] &= ~visible


def _copy_visual_mesh_shapes(
    source: newton.ModelBuilder,
    target: newton.ModelBuilder,
    *,
    body_map: dict[int, int],
    body_xforms: dict[int, wp.transform] | None = None,
    label_prefix: str = "visual",
) -> list[int]:
    """Copy visible source meshes into another builder without collision."""
    body_xforms = body_xforms or {}
    cfg = newton.ModelBuilder.ShapeConfig(
        density=0.0,
        has_shape_collision=False,
        has_particle_collision=False,
        is_visible=True,
    )
    copied: list[int] = []
    for shape, source_body in enumerate(source.shape_body):
        if source.shape_type[shape] != int(newton.GeoType.MESH):
            continue
        if not source.shape_flags[shape] & int(newton.ShapeFlags.VISIBLE):
            continue
        if source_body not in body_map:
            continue

        xform = body_xforms.get(source_body, wp.transform_identity()) * source.shape_transform[shape]
        copied.append(
            target.add_shape_mesh(
                body_map[source_body],
                xform=xform,
                mesh=source.shape_source[shape],
                scale=source.shape_scale[shape],
                cfg=cfg,
                color=source.shape_color[shape],
                label=f"{label_prefix}/{source.shape_label[shape]}",
            )
        )
    return copied


def _normalize_rgb(color) -> wp.vec3 | None:
    """Convert a trimesh material color to normalized RGB."""
    if color is None:
        return None
    values = np.asarray(color, dtype=np.float32).reshape(-1)
    if len(values) < 3:
        return None
    rgb = values[:3]
    if np.max(rgb) > 1.0:
        rgb = rgb / 255.0
    return wp.vec3(*(float(value) for value in rgb))


def _rgba_texture(image) -> np.ndarray:
    """Return a contiguous RGBA texture for Newton's ray tracer."""
    texture = np.asarray(image)
    if texture.ndim != 3 or texture.shape[2] not in (3, 4):
        raise ValueError(f"Expected an RGB or RGBA texture, got shape {texture.shape}")
    if texture.shape[2] == 3:
        alpha_value = 255 if np.issubdtype(texture.dtype, np.integer) else 1.0
        alpha = np.full((*texture.shape[:2], 1), alpha_value, dtype=texture.dtype)
        texture = np.concatenate((texture, alpha), axis=2)
    return np.ascontiguousarray(texture)


def _trimesh_visual_properties(mesh) -> tuple[wp.vec3 | None, float | None, float | None, np.ndarray | None]:
    """Extract Newton-compatible material properties from a trimesh mesh."""
    visual = mesh.visual
    material = getattr(visual, "material", None)
    color = None
    roughness = None
    metallic = None
    texture = None
    if material is not None:
        material_color = getattr(material, "baseColorFactor", None)
        if material_color is None:
            material_color = getattr(material, "diffuse", None)
        color = _normalize_rgb(material_color)
        roughness = getattr(material, "roughnessFactor", None)
        metallic = getattr(material, "metallicFactor", None)
        image = getattr(material, "baseColorTexture", None)
        if image is None:
            image = getattr(material, "image", None)
        if image is not None:
            texture = _rgba_texture(image)
    if color is None:
        color = _normalize_rgb(getattr(visual, "main_color", None))
    if color is None:
        color = wp.vec3(1.0, 1.0, 1.0)
    return color, roughness, metallic, texture


def _trimesh_render_arrays(
    mesh,
    *,
    double_sided: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Build render arrays while preserving GLB double-sided materials."""
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
    source_uvs = getattr(mesh.visual, "uv", None)
    uvs = np.asarray(source_uvs, dtype=np.float32) if source_uvs is not None else None
    if not double_sided:
        return vertices, faces.reshape(-1), normals, uvs

    vertex_count = len(vertices)
    vertices = np.concatenate((vertices, vertices), axis=0)
    faces = np.concatenate((faces, np.flip(faces, axis=1) + vertex_count), axis=0)
    # The ray tracer disables culling, so both coplanar face copies need the
    # same shading normal to avoid noisy lighting at equal-distance hits.
    normals = np.concatenate((normals, normals), axis=0)
    if uvs is not None:
        uvs = np.concatenate((uvs, uvs), axis=0)
    return vertices, faces.reshape(-1), normals, uvs


def _load_glb_visual_builder(path: Path) -> newton.ModelBuilder:
    """Load a GLB scene into a builder while preserving its submesh materials."""
    import trimesh  # noqa: PLC0415

    scene = trimesh.load(path, force="scene")
    builder = newton.ModelBuilder()
    body = builder.add_link(label=path.stem)
    cfg = newton.ModelBuilder.ShapeConfig(
        density=0.0,
        has_shape_collision=False,
        has_particle_collision=False,
        is_visible=True,
    )
    for node_name in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph[node_name]
        mesh = scene.geometry[geometry_name].copy()
        mesh.apply_transform(transform)
        color, roughness, metallic, texture = _trimesh_visual_properties(mesh)
        material = getattr(mesh.visual, "material", None)
        double_sided = bool(getattr(material, "doubleSided", False))
        vertices, indices, normals, uvs = _trimesh_render_arrays(mesh, double_sided=double_sided)
        source = newton.Mesh(
            vertices=vertices,
            indices=indices,
            normals=normals,
            uvs=uvs,
            compute_inertia=False,
            color=color,
            roughness=roughness,
            metallic=metallic,
            texture=texture,
        )
        builder.add_shape_mesh(body, mesh=source, cfg=cfg, color=color, label=str(node_name))
    return builder


def _configure_aic_lighting(viewer) -> None:
    """Approximate the Gazebo workcell's global illumination in ViewerGL."""
    renderer = viewer.renderer
    renderer.spotlight_enabled = False
    renderer.draw_shadows = False
    renderer.shadow_radius = VIEWER.shadow_radius_m
    renderer.shadow_extents = VIEWER.shadow_extents_m
    renderer.diffuse_scale = VIEWER.diffuse_scale
    renderer.specular_scale = VIEWER.specular_scale
    renderer.exposure = VIEWER.exposure
    renderer._light_color = VIEWER.light_color_rgb
    renderer.ambient_sky = VIEWER.ambient_sky_rgb
    renderer.ambient_ground = VIEWER.ambient_ground_rgb
    renderer.sky_upper = VIEWER.sky_upper_rgb
    renderer.sky_lower = VIEWER.sky_lower_rgb
    overhead = np.array(VIEWER.sun_direction, dtype=np.float32)
    renderer._sun_direction = overhead / np.linalg.norm(overhead)


def _add_glb_visual(
    builder: newton.ModelBuilder,
    *,
    asset_dir: Path,
    relative_path: str,
    body: int,
    xform: wp.transform,
    cache: dict[Path, newton.ModelBuilder],
    label: str,
) -> list[int]:
    """Attach one cached GLB scene to a Newton body."""
    path = asset_dir / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"Missing AIC visual asset: {path}")
    source = cache.get(path)
    if source is None:
        source = _load_glb_visual_builder(path)
        cache[path] = source
    if source.body_count != 1:
        raise ValueError(f"Expected one visual staging body for {path}")
    return _copy_visual_mesh_shapes(
        source,
        builder,
        body_map={0: body},
        body_xforms={0: xform},
        label_prefix=label,
    )


def _sdf_pose(element: ET.Element | None) -> wp.transform:
    """Read an SDF XYZ/RPY pose element."""
    values = [float(value) for value in (element.text if element is not None else "0 0 0 0 0 0").split()]
    if len(values) != 6:
        raise ValueError("SDF pose must contain XYZ and RPY")
    return wp.transform(wp.vec3(*values[:3]), wp.quat_rpy(*values[3:]))


def _add_sdf_box_colliders(
    builder: newton.ModelBuilder,
    *,
    model_path: Path,
    parent_pose: wp.transform,
    link_name: str,
    label_prefix: str,
    body: int = -1,
    cfg: newton.ModelBuilder.ShapeConfig | None = None,
) -> list[int]:
    """Add box colliders for one link from a vendored SDF model."""
    root = ET.parse(model_path).getroot()
    link = root.find(f".//link[@name='{link_name}']")
    if link is None:
        raise ValueError(f"{model_path} has no link named {link_name!r}")
    link_pose = parent_pose * _sdf_pose(link.find("pose"))
    if cfg is None:
        cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0,
            has_shape_collision=True,
            has_particle_collision=False,
            is_visible=False,
        )
    shapes: list[int] = []
    for collision in link.findall("collision"):
        size = collision.findtext("geometry/box/size")
        if size is None:
            continue
        dimensions = [float(value) for value in size.split()]
        if len(dimensions) != 3:
            raise ValueError(f"{model_path} collision {collision.get('name')!r} has an invalid box size")
        shapes.append(
            builder.add_shape_box(
                body,
                xform=link_pose * _sdf_pose(collision.find("pose")),
                hx=0.5 * dimensions[0],
                hy=0.5 * dimensions[1],
                hz=0.5 * dimensions[2],
                cfg=cfg,
                label=f"{label_prefix}/{collision.get('name', 'box')}",
            )
        )
    if not shapes:
        raise ValueError(f"{model_path} link {link_name!r} has no box colliders")
    return shapes

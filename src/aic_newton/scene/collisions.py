"""Define explicit collision filtering for the task-board assembly."""

import newton

AIC_AGGREGATE_REMOVED_CONNECTOR_COLLISIONS = {
    ("LC Plug", "lc_plug_link"): frozenset(
        {
            "1af02-003-ma2_collider_cylinder",
            "1af02-003-ma2001_collider_cylinder",
            "cube_collider_box.002",
            "cube_collider_box.003",
            "cube001_collider_box.002",
            "cube001_collider_box.003",
        }
    ),
    ("SFP Module", "sfp_module_link"): frozenset(
        {
            "port_collider_box",
            "port_collider_box.001",
            "port_collider_box.002",
            "port_collider_box.003",
            "port_collider_box.004",
            "port_collider_box.005",
            "port_collider_box.006",
            "port_collider_box.007",
            "latch_collider_box",
            "latch_collider_box.001",
            "latch_collider_box.002",
            "latch_collider_box.003",
            "head_collider_box",
            "head_collider_box.001",
            "head_collider_box.002",
            "head_collider_box.003",
        }
    ),
}
AIC_INTERSECTING_SFP_BOARD_SHAPE_PAIRS = (
    ("sfp_module_collision/body_collider_box", "v1015083001_collider_box"),
    ("sfp_module_collision/body_collider_box.002", "v1015083001_collider_box"),
    ("sfp_module_collision/body_collider_box.003", "v1015083001_collider_box"),
    ("sfp_tip_collision/contact_collision", "v1015083001_collider_box"),
)


def _filter_shape_sets(
    builder: newton.ModelBuilder,
    shapes_a: list[int],
    shapes_b: list[int],
) -> None:
    """Disable collision between every pair in two shape sets."""
    for shape_a in shapes_a:
        for shape_b in shapes_b:
            builder.add_shape_collision_filter_pair(shape_a, shape_b)


def _filter_body_pair(builder: newton.ModelBuilder, body_a: int, body_b: int) -> None:
    """Disable all shape collisions between two rigid bodies."""
    shapes_a = [shape for shape, body in enumerate(builder.shape_body) if body == body_a]
    shapes_b = [shape for shape, body in enumerate(builder.shape_body) if body == body_b]
    _filter_shape_sets(builder, shapes_a, shapes_b)


def _filter_fixed_grasp_collisions(
    builder: newton.ModelBuilder,
    *,
    connector_bodies: tuple[int, int],
    gripper_bodies: tuple[int, int, int],
) -> None:
    """Let the fixed grasp constraint replace unstable connector contacts."""
    for connector_body in connector_bodies:
        for gripper_body in gripper_bodies:
            _filter_body_pair(builder, connector_body, gripper_body)


def _disable_aggregate_removed_connector_collisions(
    builder: newton.ModelBuilder,
    *,
    asset_name: str,
    link_name: str,
    shapes: list[int],
) -> None:
    """Honor collision removals from the reviewed aggregate cable model."""
    removed_names = AIC_AGGREGATE_REMOVED_CONNECTOR_COLLISIONS.get((asset_name, link_name), frozenset())
    for shape in shapes:
        if builder.shape_label[shape].rsplit("/", maxsplit=1)[-1] in removed_names:
            builder.shape_flags[shape] &= ~int(newton.ShapeFlags.COLLIDE_SHAPES)


def _unique_shape_by_suffix(
    builder: newton.ModelBuilder,
    shapes: list[int] | tuple[int, ...],
    suffix: str,
) -> int:
    """Resolve exactly one shape label by suffix."""
    matches = [shape for shape in shapes if builder.shape_label[shape].endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"Expected one shape ending in {suffix!r}, found {len(matches)}")
    return matches[0]


def _filter_intersecting_sfp_board_shape_pairs(
    builder: newton.ModelBuilder,
    *,
    sfp_body: int,
    board_shapes: tuple[int, ...],
) -> None:
    """Filter only imported SFP shapes that intersect the solid board slab."""
    sfp_shapes = [shape for shape, body in enumerate(builder.shape_body) if body == sfp_body]
    sfp_suffixes = {pair[0] for pair in AIC_INTERSECTING_SFP_BOARD_SHAPE_PAIRS}
    board_suffixes = {pair[1] for pair in AIC_INTERSECTING_SFP_BOARD_SHAPE_PAIRS}

    sfp_by_suffix = {suffix: _unique_shape_by_suffix(builder, sfp_shapes, suffix) for suffix in sfp_suffixes}
    board_by_suffix = {suffix: _unique_shape_by_suffix(builder, board_shapes, suffix) for suffix in board_suffixes}
    for sfp_suffix, board_suffix in AIC_INTERSECTING_SFP_BOARD_SHAPE_PAIRS:
        builder.add_shape_collision_filter_pair(
            sfp_by_suffix[sfp_suffix],
            board_by_suffix[board_suffix],
        )


def _filter_grasped_cable_region(
    builder: newton.ModelBuilder,
    *,
    tool_body: int,
    plug_body: int,
    cable_bodies: list[int],
) -> None:
    """Disable collisions for cable segments embedded in the grasp assembly."""
    for cable_body in cable_bodies:
        _filter_body_pair(builder, tool_body, cable_body)
        _filter_body_pair(builder, plug_body, cable_body)

from __future__ import annotations

from spine_segment.checkpoint_mapping import (
    ConversionCoverage,
    check_conversion_coverage,
    map_tf_variable_to_torch_key,
    needs_conv_kernel_transpose,
    should_ignore_tf_variable,
)


def test_map_tf_variable_to_torch_key_for_unet_block() -> None:
    name = "model/unet/contracting_layers/level3/layers/2/kernel/.ATTRIBUTES/VARIABLE_VALUE"
    assert map_tf_variable_to_torch_key(name) == "unet.contracting.3.2.weight"
    assert needs_conv_kernel_transpose(name)


def test_map_tf_variable_to_torch_key_for_heatmap_heads() -> None:
    assert (
        map_tf_variable_to_torch_key(
            "model/local_heatmaps/layers/0/bias/.ATTRIBUTES/VARIABLE_VALUE"
        )
        == "local_heatmaps.0.bias"
    )
    assert (
        map_tf_variable_to_torch_key(
            "model/spatial_heatmaps/kernel/.ATTRIBUTES/VARIABLE_VALUE"
        )
        == "spatial_heatmaps.weight"
    )


def test_map_tf_variable_to_torch_key_ignores_checkpoint_bookkeeping() -> None:
    assert map_tf_variable_to_torch_key("_CHECKPOINTABLE_OBJECT_GRAPH") is None
    assert map_tf_variable_to_torch_key("save_counter/.ATTRIBUTES/VARIABLE_VALUE") is None
    assert should_ignore_tf_variable("save_counter/.ATTRIBUTES/VARIABLE_VALUE")
    assert map_tf_variable_to_torch_key("sigmas/.ATTRIBUTES/VARIABLE_VALUE") is None
    assert should_ignore_tf_variable("sigmas/.ATTRIBUTES/VARIABLE_VALUE")
    assert map_tf_variable_to_torch_key("optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE") is None
    assert (
        map_tf_variable_to_torch_key(
            "model/unet/contracting_layers/level0/layers/0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE"
        )
        is None
    )
    assert should_ignore_tf_variable(
        "model/unet/contracting_layers/level0/layers/0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE"
    )


def test_map_tf_variable_to_torch_key_accepts_unsuffixed_variable_names() -> None:
    name = "model/unet/expanding_layers/level1/layers/0/bias"
    assert map_tf_variable_to_torch_key(name) == "unet.expanding.1.0.bias"
    assert not needs_conv_kernel_transpose(name)


def test_conversion_coverage_requires_every_target_weight() -> None:
    coverage = check_conversion_coverage(
        converted_keys={"unet.contracting.0.0.weight"},
        target_keys={
            "unet.contracting.0.0.weight",
            "unet.contracting.0.0.bias",
        },
        ignored_source_names=["save_counter/.ATTRIBUTES/VARIABLE_VALUE"],
        unmapped_source_names=[],
        shape_mismatches=[],
    )

    assert coverage == ConversionCoverage(
        converted_count=1,
        ignored_source_count=1,
        missing_target_keys=("unet.contracting.0.0.bias",),
        unmapped_source_names=(),
        shape_mismatches=(),
    )

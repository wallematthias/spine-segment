from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True, slots=True)
class MappingRule:
    pattern: re.Pattern[str]
    replacement: str
    tensor_kind: str


@dataclass(frozen=True, slots=True)
class ConversionCoverage:
    converted_count: int
    ignored_source_count: int
    missing_target_keys: tuple[str, ...]
    unmapped_source_names: tuple[str, ...]
    shape_mismatches: tuple[dict[str, object], ...]


_VARIABLE_VALUE_SUFFIX = "/.ATTRIBUTES/VARIABLE_VALUE"
_IGNORED_EXACT_NAMES = {
    "_CHECKPOINTABLE_OBJECT_GRAPH",
    "save_counter",
    "sigmas",
}
_IGNORED_PREFIXES = (
    "optimizer/",
    "save_counter/",
)
_IGNORED_SUBSTRINGS = (
    "/.OPTIMIZER_SLOT/",
)

_UNET_BLOCK_RULE = re.compile(
    r"^model/(?P<prefix>unet|scnet_local|scnet_spatial)/"
    r"(?P<block>contracting_layers|expanding_layers)/level(?P<level>\d+)/"
    r"layers/(?P<layer_index>\d+)/(?P<var>kernel|bias)$"
)
_PREDICTION_RULE = re.compile(
    r"^model/prediction/layers/0/(?P<var>kernel|bias)$"
)
_LOCAL_HEATMAP_RULE = re.compile(
    r"^model/local_heatmaps/layers/0/(?P<var>kernel|bias)$"
)
_SPATIAL_HEATMAP_RULE = re.compile(
    r"^model/spatial_heatmaps/(?P<var>kernel|bias)$"
)


def normalize_tf_variable_name(variable_name: str) -> str:
    normalized = str(variable_name).strip()
    if normalized.endswith(_VARIABLE_VALUE_SUFFIX):
        normalized = normalized[: -len(_VARIABLE_VALUE_SUFFIX)]
    return normalized


def should_ignore_tf_variable(variable_name: str) -> bool:
    normalized = normalize_tf_variable_name(variable_name)
    if normalized in _IGNORED_EXACT_NAMES:
        return True
    if any(token in normalized for token in _IGNORED_SUBSTRINGS):
        return True
    return normalized.startswith(_IGNORED_PREFIXES)


def map_tf_variable_to_torch_key(variable_name: str) -> str | None:
    if should_ignore_tf_variable(variable_name):
        return None

    normalized = normalize_tf_variable_name(variable_name)
    match = _UNET_BLOCK_RULE.match(normalized)
    if match:
        prefix = match.group("prefix")
        block = match.group("block")
        level = int(match.group("level"))
        layer_index = int(match.group("layer_index"))
        variable = "weight" if match.group("var") == "kernel" else "bias"
        conv_index = layer_index // 2
        pytorch_conv_index = conv_index * 2

        if prefix == "unet":
            base = "unet"
        elif prefix == "scnet_local":
            base = "scnet_local"
        elif prefix == "scnet_spatial":
            base = "scnet_spatial"
        else:
            return None

        block_name = "contracting" if block == "contracting_layers" else "expanding"
        return f"{base}.{block_name}.{level}.{pytorch_conv_index}.{variable}"

    match = _PREDICTION_RULE.match(normalized)
    if match:
        variable = "weight" if match.group("var") == "kernel" else "bias"
        return f"prediction.{variable}"

    match = _LOCAL_HEATMAP_RULE.match(normalized)
    if match:
        variable = "weight" if match.group("var") == "kernel" else "bias"
        return f"local_heatmaps.0.{variable}"

    match = _SPATIAL_HEATMAP_RULE.match(normalized)
    if match:
        variable = "weight" if match.group("var") == "kernel" else "bias"
        return f"spatial_heatmaps.{variable}"

    return None


def needs_conv_kernel_transpose(variable_name: str) -> bool:
    return normalize_tf_variable_name(variable_name).endswith("/kernel")


def check_conversion_coverage(
    *,
    converted_keys: set[str],
    target_keys: set[str],
    ignored_source_names: list[str],
    unmapped_source_names: list[str],
    shape_mismatches: list[dict[str, object]],
) -> ConversionCoverage:
    return ConversionCoverage(
        converted_count=len(converted_keys),
        ignored_source_count=len(ignored_source_names),
        missing_target_keys=tuple(sorted(target_keys - converted_keys)),
        unmapped_source_names=tuple(sorted(unmapped_source_names)),
        shape_mismatches=tuple(shape_mismatches),
    )

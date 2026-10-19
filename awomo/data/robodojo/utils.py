# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Recursive transformations of nested normalization statistics."""

from collections.abc import Callable
from typing import Any


def dict_apply(values: dict[str, Any], function: Callable) -> dict[str, Any]:
    """Apply a function to leaf values while preserving dictionary structure."""
    result = {}
    for key, value in values.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, function)
        else:
            result[key] = function(value)
    return result

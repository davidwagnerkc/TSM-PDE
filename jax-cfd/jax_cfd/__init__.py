# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Defines the JAX-CFD module for computational fluid dynamics."""

__version__ = '0.1.0'

# Compatibility shims for deprecated JAX APIs
import jax
import jax.numpy as jnp
import jax.core
if not hasattr(jnp, 'DeviceArray'):
  jnp.DeviceArray = jax.Array
if not hasattr(jax, 'ShapedArray'):
  jax.ShapedArray = jax.core.ShapedArray
# jax.tree_* top-level aliases were removed in jax>=0.4.26 (gone entirely by the
# modern jax used for the 5090). Restore the ones jax-cfd calls at runtime.
for _alias, _impl in (
    ('tree_map', jax.tree_util.tree_map),
    ('tree_multimap', jax.tree_util.tree_map),
    ('tree_leaves', jax.tree_util.tree_leaves),
    ('tree_flatten', jax.tree_util.tree_flatten),
    ('tree_unflatten', jax.tree_util.tree_unflatten),
):
  if not hasattr(jax, _alias):
    setattr(jax, _alias, _impl)

import jax_cfd.base

from collections.abc import Sequence
import logging
import pathlib
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self._sample_actions = nnx_utils.module_jit(model.sample_actions)
        self._sample_noise = nnx_utils.module_jit(model.sample_noise)
        self._input_transform = _transforms.compose(transforms)
        self._rng = rng or jax.random.key(0)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self.action_dim = model.action_dim
        self.action_horizon = model.action_horizon
        self._get_prefix_rep = nnx_utils.module_jit(model.get_prefix_rep)
        
        # we make output transforms optional for unnormalized actions
        actions_transform = _transforms.compose(output_transforms)
        raw_actions_transform = _transforms.compose(
            remove_transform_of_type(output_transforms, _transforms.Unnormalize)
        )
        self._output_transform = ActionTransformWrapper(
            actions_transform,
            raw_actions_transform
        )
        
    @override
    def infer(self, obs: dict, noise: jnp.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        # Make a batch and convert to jax.Array.
        if inputs["state"].ndim > 1:
            batch_size = inputs["state"].shape[0]
            def _add_batch_dim(x):
                return jnp.broadcast_to(
                    x[jnp.newaxis, ...],
                    (batch_size,) + x.shape
                )
                
            inputs = jax.tree.map(lambda x: jnp.asarray(x), inputs)
            for key in inputs:
                if key not in ["image", "state"]:
                    inputs[key] = jax.tree.map(lambda x: _add_batch_dim(x), inputs[key])
        else:
            batch_size = 1
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        # self._rng, sample_rng = jax.random.split(self._rng)
        if noise is None:
            self._rng, sample_rng = jax.random.split(self._rng)
            noise = jax.random.normal(sample_rng, (batch_size, self.action_horizon, self.action_dim))
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(_model.Observation.from_dict(inputs), noise=noise, **self._sample_kwargs),
        }
        
        # Unbatch and convert to np.ndarray.
        if batch_size == 1:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
    
        return self._output_transform(outputs)
    
    def reverse_infer(self, obs: dict, action: jnp.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        # Make a batch and convert to jax.Array.
        if inputs["state"].ndim > 1:
            batch_size = inputs["state"].shape[0]
            def _add_batch_dim(x):
                return jnp.broadcast_to(
                    x[jnp.newaxis, ...],
                    (batch_size,) + x.shape
                )
                
            inputs = jax.tree.map(lambda x: jnp.asarray(x), inputs)
            for key in inputs:
                if key not in ["image", "state"]:
                    inputs[key] = jax.tree.map(lambda x: _add_batch_dim(x), inputs[key])
        else:
            batch_size = 1
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        # self._rng, sample_rng = jax.random.split(self._rng)
        if action is None:
            print("action is None, this is not expected, please check your code")
        outputs = {
            "state": inputs["state"],
            "noise": self._sample_noise(_model.Observation.from_dict(inputs), action=action, **self._sample_kwargs),
        }

        # Unbatch and convert to np.ndarray.
        if batch_size == 1:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
    
        return outputs
    
    @override
    def get_prefix_rep(self, obs: dict):
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        inputs = jax.tree.map(lambda x: jnp.asarray(x), inputs)
        # add batch dim and broadcast for keys that are not "images" and "state"
        if inputs["state"].ndim > 1:
            batch_size = inputs["state"].shape[0]
            def _add_batch_dim(x):
                return jnp.broadcast_to(
                    x[jnp.newaxis, ...],
                    (batch_size,) + x.shape
                )
            for key in inputs:
                if key not in ["image", "state"]:
                    inputs[key] = jax.tree.map(lambda x: _add_batch_dim(x), inputs[key])
        else:
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        return self._get_prefix_rep(_model.Observation.from_dict(inputs))

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
    
    
class ActionTransformWrapper:
    def __init__(self, actions_tf, raw_actions_tf):
        self.actions_tf = actions_tf
        self.raw_actions_tf = raw_actions_tf

    def __call__(self, doc):
        return_dict = {}

        unnorm_dict = doc.copy()
        trans_unnorm_dict = self.actions_tf(unnorm_dict)
        return_dict["actions"] = trans_unnorm_dict["actions"][..., :7]  # only keep the first 7 actions
            
        norm_dict = doc.copy()
        return_dict["norm_actions"] = self.raw_actions_tf(norm_dict)["actions"]
        return return_dict
    
def remove_transform_of_type(transforms_list, target_cls):
    return [t for t in transforms_list if not isinstance(t, target_cls)]
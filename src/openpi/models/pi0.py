import dataclasses
import logging
from jaxrl2.types import Params, PRNGKey
import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override
from flax.core import unfreeze
from openpi.models import model as _model
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import time
from jax.experimental import io_callback
import time, sys, numpy as _np
try:
    # JAX 新版本
    jax_io_callback = jax.experimental.io_callback
except Exception:
    # 某些环境是 jax.io_callback
    jax_io_callback = jax.io_callback

# 每次调用 sample_guidance 时的“上一段结束时间”
_SAMPLE_GUIDANCE_TIMER = {"last": None}

def _sg_timer_cb(tag, step, token_scalar):
    """Host 回调：打印阶段耗时。
    参数都是 Device 传来的标量（必须是 array），这里转成 python 标量即可。
    必须返回一个 array（哪怕是个 0），并且形状/类型跟声明一致。
    """
    tag  = int(_np.asarray(tag))
    step = int(_np.asarray(step))
    # token_scalar 只是用来建立依赖关系的占位（不打印也不使用）
    _ = float(_np.asarray(token_scalar))  # 确保取到值，建立正确的执行时序

    labels = {
        -1: "reset",
        6:  "prefill_kvcache",
        1:  "embed_suffix",
        2:  "llm_forward",
        3:  "action_out_proj",
        4:  "guidance_value_and_grad",
        5:  "update_carry",
        7:  "final_q",
    }

    now = time.perf_counter()
    if _SAMPLE_GUIDANCE_TIMER["last"] is None or tag == -1:
        _SAMPLE_GUIDANCE_TIMER["last"] = now
        print(f"[sample_guidance] {labels.get(tag, tag)} (step={step})")
    else:
        dt_ms = (now - _SAMPLE_GUIDANCE_TIMER["last"]) * 1000.0
        print(f"[sample_guidance] {labels.get(tag, tag)} (step={step}) +{dt_ms:.2f} ms")
        _SAMPLE_GUIDANCE_TIMER["last"] = now
    sys.stdout.flush()
    # 返回一个声明好的标量（形状 ()，int32）
    return _np.array(0, dtype=_np.int32)

# 统一的返回形状/类型声明（JAX 需要提前知道）
_I32_SCALAR = jax.ShapeDtypeStruct((), jnp.int32)

def _tick(tag, step_scalar, dependency_scalar):
    dep = jax.lax.stop_gradient(dependency_scalar)
    return jax_io_callback(_sg_timer_cb, _I32_SCALAR, tag, step_scalar, dep)


logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = 48

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)


class Pi0(_model.BaseModel):
    def __init__(self, config: Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.action_horizon = config.action_horizon
        self.action_dim = config.action_dim
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
            )
        )
        llm.lazy_init(rngs=rngs, method="init")
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # add a single state token
        state_token = self.state_proj(obs.state)[:, None, :]
        tokens.append(state_token)
        input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
        # image/language inputs do not attend to state or actions
        ar_mask += [True]

        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        # mix timestep + action information using an MLP
        action_tokens = self.action_in_proj(noisy_actions)
        time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
        action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
        action_time_tokens = self.action_time_mlp_in(action_time_tokens)
        action_time_tokens = nnx.swish(action_time_tokens)
        action_time_tokens = self.action_time_mlp_out(action_time_tokens)
        tokens.append(action_time_tokens)
        input_mask.append(jnp.ones(action_time_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    # @override
    # def sample_actions(
    #     self,
    #     observation: _model.Observation,
    #     *,
    #     noise: jnp.ndarray,
    #     guidance: dict | None = None,
    #     guidance_obs: dict | None = None,
    #     guidance_scale: float = 1.0,
    #     num_steps: int | at.Int[at.Array, ""] = 10,
    # ) -> _model.Actions:
    #     observation = _model.preprocess_observation(None, observation, train=False)
    #     # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
    #     # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
    #     dt = -1.0 / num_steps
    #     batch_size = observation.state.shape[0]
    #     # first fill KV cache with a forward pass of the prefix
    #     prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
    #     prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    #     positions = jnp.cumsum(prefix_mask, axis=1) - 1
    #     _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
    #     current_denoise_step = 10
    #     def step(carry):
    #         x_t, time, current_denoise_step = carry
    #         suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(
    #             observation, x_t, jnp.broadcast_to(time, batch_size)
    #         )
    #         # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
    #         # other
    #         suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
    #         # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
    #         # prefix tokens
    #         prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
    #         # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
    #         # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
    #         full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
    #         assert full_attn_mask.shape == (
    #             batch_size,
    #             suffix_tokens.shape[1],
    #             prefix_tokens.shape[1] + suffix_tokens.shape[1],
    #         )
    #         # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
    #         positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

    #         (prefix_out, suffix_out), _ = self.PaliGemma.llm(
    #             [None, suffix_tokens], mask=full_attn_mask, positions=positions, kv_cache=kv_cache
    #         )
    #         assert prefix_out is None
    #         v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

    #         # with guidance
    #         single_action = jax.lax.stop_gradient(x_t[:,0,:])  # shape [B, action_dim]
    #         # calculate the gradient of q_value w.r.t. single_action
    #         grad_fn = jax.jit(jax.grad(
    #             lambda a: guidance.get_q_values(guidance_obs, a, current_denoise_step)
    #         ))
    #         q_grad = grad_fn(single_action)
    #         # print('q_grad:', q_grad)
    #         # print('q_value:', q_value)
    #         v_t = v_t + guidance_scale * q_grad[None, :]
    #         current_denoise_step -= 1 

    #         return x_t + dt * v_t, time + dt, current_denoise_step

    #     def cond(carry):
    #         x_t, time, ts = carry
    #         # robust to floating-point error
    #         return time >= -dt / 2
    #     x_0, _, _ = jax.lax.while_loop(cond, step, (noise, 1.0, current_denoise_step))
    #     return x_0
    
    
    @override
    def sample_actions(
        self,
        observation: _model.Observation,
        *,
        noise: jnp.ndarray,
        guidance: dict | None = None,
        guidance_obs: dict | None = None,
        guidance_scale: float = 1.0,
        num_steps: int = 10,
    ) -> tuple[_model.Actions, list[jnp.ndarray]]:
        observation = _model.preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
        
        def get_q_sum_fn(action, time_index):
            # time array
            times = jnp.full((batch_size, 1), time_index, dtype=jnp.int32)
            q_value = guidance.apply_fn({'params': guidance.params}, guidance_obs, action, times)
            q_sum = jnp.sum(q_value) # ()
            return q_sum

        current_denoise_step = num_steps

        def step_fn(carry, _):
            x_t, time, current_denoise_step = carry
            suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)

            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens], mask=full_attn_mask, positions=positions, kv_cache=kv_cache
            )
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            # # ---- with guidance ----
            # single_action = jax.lax.stop_gradient(x_t[:, 0, :])  # shape [B, action_dim]
    
            # # only grad
            # grad_fn = jax.grad(lambda a: get_q_sum_fn(a, current_denoise_step))
            # q_grad = grad_fn(single_action)

            # # val_and_grad_fn = jax.value_and_grad(get_q_sum_fn, has_aux=True)
            # # q_sum, q_grad = val_and_grad_fn(single_action)

            # q_grad = jnp.broadcast_to(q_grad[:, None, :], v_t.shape)  # (B, 50, 32)
            # v_t = v_t + guidance_scale * q_grad

            # 下一步 carrydef q_sum_fn(a, t, guidance_obs, guidance_params):
            new_carry = (x_t + dt * v_t, time + dt, current_denoise_step - 1)

            # 这里直接把 q_means 返回，scan 会自动收集
            return new_carry, x_t[:, 0, :]  # (B, action_dim)

        # 扫描 num_steps 次，收集 q_seq
        (x_0, _, _), a_seq = jax.lax.scan(
            step_fn,
            (noise, 1.0, current_denoise_step),
            xs=None,
            length=num_steps,
        )  # a_seq.shape = (num_steps, batch_size)

        # 把 final_q 拼到 a_seq 最后，并在第0维反转
        a_list = jnp.concatenate([a_seq, x_0[:, 0, :][None]], axis=0)  # (num_steps+1, batch_size)
        a_list = jnp.flip(a_list, axis=0).transpose(1, 0, 2)  # (batch_size, num_steps+1, action_dim)

        # print(f"length of q_list: {len(q_list)}, shape of each q: {q_list[0].shape}")

        return x_0, a_list
    

    def sample_actions_train(
        self,
        observation: _model.Observation,
        *,
        noise: jnp.ndarray,
        guidance: dict | None = None,
        guidance_obs: dict | None = None,
        guidance_scale: float = 1.0,
        num_steps: int = 10,
    ) -> tuple[_model.Actions, list[jnp.ndarray]]:
        observation = unfreeze(observation)   # 先解冻
        observation = _model.Observation.from_dict(observation)
        observation = _model.preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
        
        def get_q_sum_fn(action, time_index):
            # time array
            times = jnp.full((batch_size, 1), time_index, dtype=jnp.int32)
            q_value = guidance.apply_fn({'params': guidance.params}, guidance_obs, action, times)
            q_sum = jnp.sum(q_value) # ()
            return q_sum

        current_denoise_step = num_steps

        def step_fn(carry, _):
            x_t, time, current_denoise_step = carry
            suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)

            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens], mask=full_attn_mask, positions=positions, kv_cache=kv_cache
            )
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            # # ---- with guidance ----
            # single_action = jax.lax.stop_gradient(x_t[:, 0, :])  # shape [B, action_dim]
    
            # # only grad
            # grad_fn = jax.grad(lambda a: get_q_sum_fn(a, current_denoise_step))
            # q_grad = grad_fn(single_action)

            # # val_and_grad_fn = jax.value_and_grad(get_q_sum_fn, has_aux=True)
            # # q_sum, q_grad = val_and_grad_fn(single_action)

            # q_grad = jnp.broadcast_to(q_grad[:, None, :], v_t.shape)  # (B, 50, 32)
            # v_t = v_t + guidance_scale * q_grad

            # 下一步 carrydef q_sum_fn(a, t, guidance_obs, guidance_params):
            new_carry = (x_t + dt * v_t, time + dt, current_denoise_step - 1)

            # 这里直接把 q_means 返回，scan 会自动收集
            return new_carry, x_t[:, 0, :]  # (B, action_dim)

        # 扫描 num_steps 次，收集 q_seq
        (x_0, _, _), a_seq = jax.lax.scan(
            step_fn,
            (noise, 1.0, current_denoise_step),
            xs=None,
            length=num_steps,
        )  # a_seq.shape = (num_steps, batch_size)

        # 把 final_q 拼到 a_seq 最后，并在第0维反转
        a_list = jnp.concatenate([a_seq, x_0[:, 0, :][None]], axis=0)  # (num_steps+1, batch_size)
        a_list = jnp.flip(a_list, axis=0).transpose(1, 0, 2)  # (batch_size, num_steps+1, action_dim)

        # print(f"length of q_list: {len(q_list)}, shape of each q: {q_list[0].shape}")

        return x_0[:, 0, :]
    

    def sample_actions_w_guidance(
        self,
        observation: _model.Observation,
        *,
        noise: jnp.ndarray,
        guidance: dict | None = None,
        guidance_obs: dict | None = None,
        guidance_scale: float = 1.0,
        use_guidance: bool = True,
        num_steps: int = 10,
    ) -> tuple[_model.Actions, list[jnp.ndarray]]:
        observation = _model.preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
        
        def get_q_sum_fn(action, time_index):
            # time array
            times = jnp.full((batch_size, 1), time_index, dtype=jnp.int32)
            q_value = guidance.apply_fn({'params': guidance.params}, guidance_obs, action, times)
            q_sum = jnp.sum(q_value) # ()
            return q_sum

        current_denoise_step = num_steps

        def step_fn(carry, _):
            x_t, time, current_denoise_step = carry
            suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)

            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens], mask=full_attn_mask, positions=positions, kv_cache=kv_cache
            )
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            if use_guidance:
                # ---- with guidance ----
                single_action = jax.lax.stop_gradient(x_t[:, 0, :])  # shape [B, action_dim]
        
                # only grad
                grad_fn = jax.grad(lambda a: get_q_sum_fn(a, current_denoise_step))
                q_grad = grad_fn(single_action)

                # val_and_grad_fn = jax.value_and_grad(get_q_sum_fn, has_aux=True)
                # q_sum, q_grad = val_and_grad_fn(single_action)

                q_grad = jnp.broadcast_to(q_grad[:, None, :], v_t.shape)  # (B, 50, 32)
                v_t = v_t + guidance_scale * q_grad

            # 下一步 carrydef q_sum_fn(a, t, guidance_obs, guidance_params):
            new_carry = (x_t + dt * v_t, time + dt, current_denoise_step - 1)

            # 这里直接把 q_means 返回，scan 会自动收集
            return new_carry, x_t[:, 0, :]  # (B, action_dim)

        # 扫描 num_steps 次，收集 q_seq
        (x_0, _, _), a_seq = jax.lax.scan(
            step_fn,
            (noise, 1.0, current_denoise_step),
            xs=None,
            length=num_steps,
        )  # a_seq.shape = (num_steps, batch_size)

        # 把 final_q 拼到 a_seq 最后，并在第0维反转
        a_list = jnp.concatenate([a_seq, x_0[:, 0, :][None]], axis=0)  # (num_steps+1, batch_size)
        a_list = jnp.flip(a_list, axis=0).transpose(1, 0, 2)  # (batch_size, num_steps+1, action_dim)

        # print(f"length of q_list: {len(q_list)}, shape of each q: {q_list[0].shape}")

        return x_0, a_list
    

    def sample_actions_w_guidance_train(
        self,
        observation: _model.Observation,
        *,
        noise: jnp.ndarray,
        guidance: dict | None = None,
        guidance_obs: dict | None = None,
        guidance_scale: float = 1.0,
        num_steps: int = 10,
    ) -> tuple[_model.Actions, list[jnp.ndarray]]:
        observation = unfreeze(observation)   # 先解冻
        observation = _model.Observation.from_dict(observation)
        observation = _model.preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
        
        def get_q_sum_fn(action, time_index):
            # time array
            times = jnp.full((batch_size, 1), time_index, dtype=jnp.int32)
            q_value = guidance['apply_fn']({'params': guidance['params']}, guidance_obs, action, times)
            q_sum = jnp.sum(q_value) # ()
            return q_sum

        current_denoise_step = num_steps

        def step_fn(carry, _):
            x_t, time, current_denoise_step = carry
            suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)

            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens], mask=full_attn_mask, positions=positions, kv_cache=kv_cache
            )
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            # ---- with guidance ----
            single_action = jax.lax.stop_gradient(x_t[:, 0, :])  # shape [B, action_dim]
    
            # only grad
            grad_fn = jax.grad(lambda a: get_q_sum_fn(a, current_denoise_step))
            q_grad = grad_fn(single_action)

            # val_and_grad_fn = jax.value_and_grad(get_q_sum_fn, has_aux=True)
            # q_sum, q_grad = val_and_grad_fn(single_action)

            q_grad = jnp.broadcast_to(q_grad[:, None, :], v_t.shape)  # (B, 50, 32)
            v_t = v_t + guidance_scale * q_grad

            # 下一步 carrydef q_sum_fn(a, t, guidance_obs, guidance_params):
            new_carry = (x_t + dt * v_t, time + dt, current_denoise_step - 1)

            # 这里直接把 q_means 返回，scan 会自动收集
            return new_carry, x_t[:, 0, :]  # (B, action_dim)

        # 扫描 num_steps 次，收集 q_seq
        (x_0, _, _), a_seq = jax.lax.scan(
            step_fn,
            (noise, 1.0, current_denoise_step),
            xs=None,
            length=num_steps,
        )  # a_seq.shape = (num_steps, batch_size)

        # 把 final_q 拼到 a_seq 最后，并在第0维反转
        a_list = jnp.concatenate([a_seq, x_0[:, 0, :][None]], axis=0)  # (num_steps+1, batch_size)
        a_list = jnp.flip(a_list, axis=0).transpose(1, 0, 2)  # (batch_size, num_steps+1, action_dim)

        # print(f"length of q_list: {len(q_list)}, shape of each q: {q_list[0].shape}")

        return x_0[:, 0, :]
    

    def sample_guidance(
        self,
        observation: _model.Observation,
        *,
        noise: jnp.ndarray,
        guidance: dict | None = None,
        guidance_obs: dict | None = None,
        guidance_scale: float = 1.0,
        num_steps: int = 10,
    ) -> tuple[_model.Actions, list[jnp.ndarray]]:
        # print("observation obs keys:", observation.keys())
        observation = unfreeze(observation)   # 先解冻
        observation = _model.Observation.from_dict(observation)
        observation = _model.preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
        
        def get_q_sum_fn(action):
            # time array
            B = action.shape[0]
            times = jnp.full((B, 1), current_denoise_step, dtype=jnp.int32)

            input_collections = {'params': guidance['params']}
            q_value = guidance['apply_fn'](input_collections, guidance_obs, action, times)
            q_value = jnp.swapaxes(q_value, 0, 1)  # (num_qs, B)
            q_sum = jnp.sum(q_value) # ()
            q_means = jnp.mean(q_value, axis=-1) # (B,)
            return q_sum, q_means

        current_denoise_step = num_steps

        def step_fn(carry, _):
            x_t, time, current_denoise_step = carry
            suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)

            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens], mask=full_attn_mask, positions=positions, kv_cache=kv_cache
            )
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            # ---- with guidance ----
            single_action = jax.lax.stop_gradient(x_t[:, 0, :])  # shape [B, action_dim]
    
            # val_and_grad_fn = jax.value_and_grad(get_q_sum_fn, has_aux=True)
            # (q_sum, q_means), q_grad = val_and_grad_fn(single_action)
            
            q_means = get_q_sum_fn(single_action)[1]  # (B,)

            # # === 关键：实时打印 ===
            # jax.debug.print("q_grad shape: {x}, values: {y}", 
            #                 x=q_grad.shape, 
            #                 y=q_grad)

            # q_grad = jnp.broadcast_to(q_grad[:, None, :], v_t.shape)  # (B, 50, 32)
            # v_t = v_t + guidance_scale * q_grad

            # 下一步 carrydef q_sum_fn(a, t, guidance_obs, guidance_params):

            new_carry = (x_t + dt * v_t, time + dt, current_denoise_step - 1)

            # 这里直接把 q_means 返回，scan 会自动收集
            return new_carry, q_means

        # 扫描 num_steps 次，收集 q_seq
        (x_0, _, _), q_seq = jax.lax.scan(
            step_fn,
            (noise, 1.0, current_denoise_step),
            xs=None,
            length=num_steps,
        )  # q_seq.shape = (num_steps, batch_size)

        # 最后再算一次 Q
        final_action = jax.lax.stop_gradient(x_0[:, 0, :])
        # _, final_q = guidance.get_q_sum(guidance_obs, final_action, 0, guidance_params)
        
        input_collections = {'params': guidance['params']}
        B = final_action.shape[0]
        times = jnp.full((B, 1), current_denoise_step, dtype=jnp.int32)
        final_q = guidance['apply_fn'](input_collections, guidance_obs, final_action, times)
        final_q = jnp.swapaxes(final_q, 0, 1)  # (num_qs, B)
        final_q = jnp.mean(final_q, axis=-1) # (B,)

        # 把 final_q 拼到 q_seq 最后，并在第0维反转
        q_list = jnp.concatenate([q_seq, final_q[None]], axis=0)  # (num_steps+1, batch_size)
        q_list = jnp.flip(q_list, axis=0)  # 反转

        # print(f"length of q_list: {len(q_list)}, shape of each q: {q_list[0].shape}")

        return q_list
    
    
    
    # def sample_guidance(
    #     self,
    #     observation: _model.Observation,
    #     *,
    #     noise: jnp.ndarray,
    #     guidance: dict | None = None,
    #     guidance_obs: dict | None = None,
    #     guidance_scale: float = 1.0,
    #     guidance_params: Params = None,
    #     num_steps: int | at.Int[at.Array, ""] = 10,
    # ) -> tuple[_model.Actions, list[jnp.ndarray]]:
    #     observation = _model.preprocess_observation(None, observation, train=False)
    #     dt = -1.0 / num_steps
    #     batch_size = observation.state.shape[0]

    #     # first fill KV cache with a forward pass of the prefix
    #     prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
    #     prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    #     positions = jnp.cumsum(prefix_mask, axis=1) - 1
    #     _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

    #     current_denoise_step = num_steps

    #     def step(carry, _):
    #         x_t, time, current_denoise_step = carry
    #         suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(
    #             observation, x_t, jnp.broadcast_to(time, batch_size)
    #         )
    #         suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
    #         prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
    #         full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)

    #         positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
    #         (prefix_out, suffix_out), _ = self.PaliGemma.llm(
    #             [None, suffix_tokens], mask=full_attn_mask, positions=positions, kv_cache=kv_cache
    #         )
    #         v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

    #         # ---- with guidance ----
    #         single_action = jax.lax.stop_gradient(x_t[:, 0, :])  # shape [B, action_dim]

    #         def loss_and_aux(a):
    #             q_sum, q_means = guidance.get_q_sum(
    #                 guidance_obs, a, current_denoise_step, guidance_params
    #             )
    #             return q_sum, q_means   # q_sum 用来算梯度，q_means 当 aux

    #         val_and_grad_fn = jax.jit(
    #             jax.value_and_grad(loss_and_aux, has_aux=True)
    #         )
    #         (q_sum, q_means), q_grad = val_and_grad_fn(single_action)

    #         q_grad = jnp.broadcast_to(q_grad[:, None, :], v_t.shape)  # (B, 50, 32)
    #         v_t = v_t + guidance_scale * q_grad

    #         # 下一步 carry
    #         new_carry = (x_t + dt * v_t, time + dt, current_denoise_step - 1)

    #         # 这里直接把 q_means 返回，scan 会自动收集
    #         return new_carry, q_means

    #     # 扫描 num_steps 次，收集 q_seq
    #     (x_0, _, _), q_seq = jax.lax.scan(
    #         step,
    #         (noise, 1.0, current_denoise_step),
    #         xs=None,
    #         length=num_steps,
    #     )  # q_seq.shape = (num_steps, batch_size)

    #     # 最后再算一次 Q
    #     final_action = jax.lax.stop_gradient(x_0[:, 0, :])
    #     _, final_q = guidance.get_q_sum(guidance_obs, final_action, 0, guidance_params)

    #     # 把 final_q 拼到 q_seq 最后，并在第0维反转
    #     q_list = jnp.concatenate([q_seq, final_q[None]], axis=0)  # (num_steps+1, batch_size)
    #     q_list = jnp.flip(q_list, axis=0)  # 反转

    #     # print(f"length of q_list: {len(q_list)}, shape of each q: {q_list[0].shape}")

    #     return x_0, q_list
    
        
    # @override
    # def sample_guidance(
    #     self,
    #     observation: _model.Observation,
    #     *,
    #     noise: jnp.ndarray,
    #     guidance: dict | None = None,
    #     guidance_obs: dict | None = None,
    #     guidance_scale: float = 1.0,
    #     guidance_params: Params = None,
    #     num_steps: int | at.Int[at.Array, ""] = 10,
    # ) -> tuple[_model.Actions, list[jnp.ndarray]]:
    #     observation = _model.preprocess_observation(None, observation, train=False)
    #     dt = -1.0 / num_steps
    #     batch_size = observation.state.shape[0]

    #     # ---- reset 打点（不依赖任何重计算）----
    #     _ = _tick(jnp.int32(-1), jnp.int32(num_steps), jnp.array(0.0, dtype=jnp.float32))

    #     # ---- 1) prefill KV cache ----
    #     prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
    #     prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    #     positions = jnp.cumsum(prefix_mask, axis=1) - 1
    #     # 拿到 prefix_out 以便建立依赖（方便“prefill”打点发生在 LLM 之后）
    #     (prefix_out, _), kv_cache = self.PaliGemma.llm(
    #         [prefix_tokens, None], mask=prefix_attn_mask, positions=positions
    #     )
    #     # prefill 打点：依赖 prefix_out 的 reduce 标量，避免搬大张量到 host
    #     _ = _tick(jnp.int32(6), jnp.int32(num_steps), jnp.sum(prefix_out))

    #     current_denoise_step = num_steps

    #     def step(carry, _):
    #         x_t, time_s, cur_step = carry

    #         # ---- 2) embed_suffix ----
    #         suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(
    #             observation, x_t, jnp.broadcast_to(time_s, batch_size)
    #         )
    #         suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
    #         prefix_attn_mask2 = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
    #         full_attn_mask = jnp.concatenate([prefix_attn_mask2, suffix_attn_mask], axis=-1)
    #         # 打点：依赖 suffix_mask 的 sum（很小），确保发生在 embed_suffix 之后
    #         _ = _tick(jnp.int32(1), jnp.int32(cur_step), jnp.sum(suffix_mask))

    #         # ---- 3) llm_forward （带 kv_cache）----
    #         positions2 = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
    #         (prefix_out2, suffix_out), _ = self.PaliGemma.llm(
    #             [None, suffix_tokens], mask=full_attn_mask, positions=positions2, kv_cache=kv_cache
    #         )
    #         _ = _tick(jnp.int32(2), jnp.int32(cur_step), jnp.sum(suffix_out))

    #         # ---- 4) action_out_proj ----
    #         v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])
    #         _ = _tick(jnp.int32(3), jnp.int32(cur_step), jnp.sum(v_t))

    #         # ---- 5) guidance: value_and_grad w.r.t action ----
    #         single_action = jax.lax.stop_gradient(x_t[:, 0, :])  # [B, action_dim]

    #         def loss_and_aux(a):
    #             q_sum, q_means = guidance.get_q_sum(guidance_obs, a, cur_step, guidance_params)
    #             return q_sum, q_means  # q_sum for grad; q_means as aux

    #         val_and_grad_fn = jax.jit(jax.value_and_grad(loss_and_aux, has_aux=True))
    #         (q_sum, q_means), q_grad = val_and_grad_fn(single_action)
    #         _ = _tick(jnp.int32(4), jnp.int32(cur_step), q_sum)  # q_sum 是标量，很合适做依赖

    #         # 将梯度广播到 v_t 形状并应用
    #         q_grad = jnp.broadcast_to(q_grad[:, None, :], v_t.shape)  # (B, H, D)
    #         v_t = v_t + guidance_scale * q_grad

    #         # ---- 6) update_carry ----
    #         new_x = x_t + dt * v_t
    #         new_time = time_s + dt
    #         new_step = cur_step - 1
    #         _ = _tick(jnp.int32(5), jnp.int32(cur_step), jnp.sum(new_x))

    #         # 返回新的 carry 以及 aux（q_means 会被 scan 收集）
    #         return (new_x, new_time, new_step), q_means

    #     # 扫描 num_steps 次，收集 q_seq
    #     (x_0, _, _), q_seq = jax.lax.scan(
    #         step,
    #         (noise, 1.0, current_denoise_step),
    #         xs=None,
    #         length=num_steps,
    #     )

    #     # ---- 7) final_q ----
    #     final_action = jax.lax.stop_gradient(x_0[:, 0, :])
    #     _, final_q = guidance.get_q_sum(guidance_obs, final_action, 0, guidance_params)
    #     _ = _tick(jnp.int32(7), jnp.int32(0), jnp.mean(final_q))

    #     # 拼接并反转时间维
    #     q_list = jnp.concatenate([q_seq, final_q[None]], axis=0)
    #     q_list = jnp.flip(q_list, axis=0)

    #     return x_0, q_list


    
    def sample_noise(
        self,
        observation: _model.Observation,
        *,
        action: jnp.ndarray,
        num_steps: int | at.Int[at.Array, ""] = 10,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = 1.0 / num_steps
        batch_size = observation.state.shape[0]
        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens], mask=full_attn_mask, positions=positions, kv_cache=kv_cache
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time <= 1.0 - dt / 2
        x_1, _ = jax.lax.while_loop(cond, step, (action, 0.0))
        return x_1
    
    # get the prfix representation
    def get_prefix_rep(self, observation: _model.Observation):
        """
        Returns the Gemma (VLM) hidden‐state representations for images + language.
        Output shape is [B, S_prefix, W], where:
          B = batch size,
          S_prefix = total # of image tokens + text tokens,
          W = Gemma hidden‑width.
        """
        observation = _model.preprocess_observation(None, observation, train=False)
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (hidden_state, _), kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
        return hidden_state, kv_cache

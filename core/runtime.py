"""FLUX.2 Klein runtime: CFG-aware velocity, ODE prefix/tail, optional one-step SDE."""

from __future__ import annotations

import gc
from contextlib import contextmanager
from typing import Any, Iterator

from .lora import LoRAManager, LoraMode
from .teacher import propose_teacher
from .transition import Transition, apply_transition, sde_mean_from_velocity


class Flux2Runtime:
    def __init__(self, cfg: dict[str, Any]):
        import accelerate
        import diffusers
        import torch
        import transformers

        self.torch = torch
        self.cfg = cfg
        model_cfg = cfg["model"]
        self.guidance = float(model_cfg.get("guidance", 4.0))
        self.height = int(model_cfg.get("height", 1024))
        self.width = int(model_cfg.get("width", 1024))
        self.steps = int(model_cfg.get("steps", 50))
        self.pipe = diffusers.Flux2KleinPipeline.from_pretrained(
            model_cfg["path"],
            torch_dtype=torch.bfloat16,
            local_files_only=bool(model_cfg.get("local_files_only", True)),
        )
        self.pipe.to("cuda")
        self.pipe.set_progress_bar_config(disable=True)
        self.pipe._guidance_scale = self.guidance
        self.pipe._attention_kwargs = None
        vae = getattr(self.pipe, "vae", None)
        if vae is not None:
            if hasattr(vae, "enable_slicing"):
                vae.enable_slicing()
            if hasattr(vae, "enable_tiling"):
                vae.enable_tiling()
        if model_cfg.get("gradient_checkpointing", True) and hasattr(self.pipe.transformer, "enable_gradient_checkpointing"):
            self.pipe.transformer.enable_gradient_checkpointing()
        self.device = self.pipe._execution_device
        self.lora = LoRAManager(self.pipe.transformer, cfg)
        self.prompt_cache: dict[str, tuple[Any, Any]] = {}
        self.timesteps = None
        self._latent_ids = None
        self.environment = {
            "torch": torch.__version__,
            "diffusers": diffusers.__version__,
            "transformers": transformers.__version__,
            "accelerate": accelerate.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "pipeline": type(self.pipe).__name__,
            "n_lora": self.lora.n_trainable(),
        }
        sde_cfg = cfg.get("sde", {})
        self.eta = float(sde_cfg.get("eta", sde_cfg.get("noise_coeff", 0.3)))
        self.noise_coeff = self.eta
        self.next_cap = float(sde_cfg.get("std_cap_on_next", 0.95))
        self.min_std = float(sde_cfg.get("min_std", 1e-6))
        self.decode_offload = bool(model_cfg.get("decode_offload_transformer", True))
        self.gamma = float(cfg.get("teacher_guidance_scale", 1.0))

    @contextmanager
    def lora_mode(self, mode: LoraMode) -> Iterator[None]:
        previous = self.lora.layers[0].mode if self.lora.layers else "student"
        self.lora.set_mode(mode)
        try:
            yield
        finally:
            self.lora.set_mode(previous)

    def snapshot_old(self) -> None:
        self.lora.snapshot_old()

    def encode(self, prompt: str) -> tuple[Any, Any]:
        if prompt not in self.prompt_cache:
            with self.torch.inference_mode():
                self.prompt_cache[prompt] = self.pipe.encode_prompt(
                    prompt=prompt, device=self.device, num_images_per_prompt=1
                )
        return self.prompt_cache[prompt]

    def precompute_prompts(self, texts: list[str]) -> None:
        for text in texts:
            self.encode(text)
        if self.cfg["model"].get("offload_text_encoder_after_precompute", True):
            encoder = getattr(self.pipe, "text_encoder", None)
            if encoder is not None:
                encoder.to("cpu")
                self.torch.cuda.empty_cache()

    def prepare_latents(self, seed: int):
        torch = self.torch
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        num_channels = self.pipe.transformer.config.in_channels // 4
        return self.pipe.prepare_latents(
            batch_size=1,
            num_latents_channels=num_channels,
            height=self.height,
            width=self.width,
            dtype=torch.bfloat16,
            device=self.device,
            generator=generator,
        )

    def reset_schedule(self, image_seq_len: int):
        import numpy as np
        from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu, retrieve_timesteps

        sigmas = np.linspace(1.0, 1 / self.steps, self.steps)
        if getattr(self.pipe.scheduler.config, "use_flow_sigmas", False):
            sigmas = None
        mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=self.steps)
        timesteps, _ = retrieve_timesteps(
            self.pipe.scheduler, self.steps, self.device, sigmas=sigmas, mu=mu,
        )
        if hasattr(self.pipe.scheduler, "set_begin_index"):
            self.pipe.scheduler.set_begin_index(0)
        self.timesteps = timesteps
        return timesteps

    def sigma_pair(self, step: int) -> tuple[Any, Any, Any]:
        scheduler = self.pipe.scheduler
        t = self.timesteps[step]
        sigma = scheduler.sigmas[step]
        sigma_next = scheduler.sigmas[step + 1]
        return t, sigma, sigma_next

    def _transformer(self, latents, timestep, embeds, text_ids, latent_ids):
        return self.pipe.transformer(
            hidden_states=latents.to(self.pipe.transformer.dtype),
            timestep=timestep / 1000,
            guidance=None,
            encoder_hidden_states=embeds,
            txt_ids=text_ids,
            img_ids=latent_ids,
            joint_attention_kwargs=self.pipe.attention_kwargs,
            return_dict=False,
        )[0][:, : latents.size(1) :]

    def predict_guided(
        self,
        latents,
        step: int,
        prompt: str,
        cond_mode: LoraMode,
        *,
        grad: bool = False,
    ):
        """CFG: LoRA only on the conditional branch. Uncond is always frozen Base."""
        torch = self.torch
        t, _, _ = self.sigma_pair(step)
        timestep = t.expand(latents.shape[0]).to(latents.dtype)
        cond_embeds, cond_ids = self.encode(prompt)
        uncond_embeds, uncond_ids = self.encode("")
        latent_ids = self._latent_ids
        context = torch.enable_grad() if grad else torch.no_grad()
        with context:
            with self.lora_mode(cond_mode):
                cond = self._transformer(latents, timestep, cond_embeds, cond_ids, latent_ids)
            with self.lora_mode("base"):
                if grad:
                    with torch.no_grad():
                        uncond = self._transformer(latents, timestep, uncond_embeds, uncond_ids, latent_ids)
                else:
                    uncond = self._transformer(latents, timestep, uncond_embeds, uncond_ids, latent_ids)
            return uncond + self.guidance * (cond - uncond)

    def predict_conditional(
        self,
        latents,
        step: int,
        prompt: str,
        cond_mode: LoraMode,
        *,
        grad: bool = False,
    ):
        """Conditional transformer only. Used for structure residual: Δ = g (c_S - c_P)."""
        torch = self.torch
        t, _, _ = self.sigma_pair(step)
        timestep = t.expand(latents.shape[0]).to(latents.dtype)
        cond_embeds, cond_ids = self.encode(prompt)
        latent_ids = self._latent_ids
        context = torch.enable_grad() if grad else torch.no_grad()
        with context:
            with self.lora_mode(cond_mode):
                return self._transformer(latents, timestep, cond_embeds, cond_ids, latent_ids)

    def structure_residual(self, latents, step: int, plain: str, structured: str):
        """g * (c_struct - c_plain). Equals v_struct - v_plain because uncond cancels."""
        with self.torch.no_grad():
            c_plain = self.predict_conditional(latents, step, plain, "base", grad=False)
            c_struct = self.predict_conditional(latents, step, structured, "base", grad=False)
        return self.guidance * (c_struct - c_plain)

    def teacher_velocity_from_old(self, v_old, latents, step: int, plain: str, structured: str):
        return v_old + self.gamma * self.structure_residual(latents, step, plain, structured)

    def run_teacher_structured_ode(self, latents, v_old, step: int, plain: str, structured: str):
        """Branch residual as Euler ODE, then finish with the structured prompt.

        No teacher SDE. Distillation still targets this first Euler mean; the
        remaining steps use structured CFG so the final image can keep the layout.
        """
        v_teacher = self.teacher_velocity_from_old(v_old, latents, step, plain, structured)
        teacher = self.apply_kernel(latents, v_teacher, step, use_sde=False)
        next_latents, _ = self.finish_tail(teacher.next_latents, structured, step + 1, "old")
        return teacher, v_teacher, next_latents

    def apply_kernel(self, latents, velocity, step: int, *, use_sde: bool, noise=None, generator=None) -> Transition:
        _, sigma, sigma_next = self.sigma_pair(step)
        return apply_transition(
            latents,
            velocity,
            sigma,
            sigma_next,
            use_sde=use_sde,
            noise=noise,
            generator=generator,
            noise_coeff=self.noise_coeff,
            eta=self.eta,
            next_cap=self.next_cap,
            min_std=self.min_std,
        )

    def sde_mean(self, latents, velocity, step: int, std=None):
        _, sigma, sigma_next = self.sigma_pair(step)
        return sde_mean_from_velocity(latents, velocity, sigma, sigma_next, std=std)

    def run_ode_range(self, latents, prompt: str, start: int, end: int, mode: LoraMode, keep_steps=()):
        keep = []
        for step in range(start, end):
            velocity = self.predict_guided(latents, step, prompt, mode, grad=False)
            if step in keep_steps:
                keep.append(
                    {
                        "step": step,
                        "latents": latents.detach().to("cpu"),
                        "velocity": velocity.detach().to("cpu"),
                        "timestep": float(self.timesteps[step].detach().float().cpu()),
                    }
                )
            latents = self.apply_kernel(latents, velocity, step, use_sde=False).next_latents
        return latents, keep

    def prefix_to_branch(self, seed: int, prompt: str, n_completed: int, mode: LoraMode = "old"):
        latents, latent_ids = self.prepare_latents(seed)
        self._latent_ids = latent_ids
        self.reset_schedule(latents.shape[1])
        latents, _ = self.run_ode_range(latents, prompt, 0, n_completed, mode)
        step = n_completed
        t, sigma, sigma_next = self.sigma_pair(step)
        return {
            "latents": latents,
            "latent_ids": latent_ids,
            "step": step,
            "timestep": t,
            "sigma": sigma,
            "sigma_next": sigma_next,
        }

    def teacher_guided_velocity(self, latents, step: int, plain: str, structured: str):
        """Eager path kept for identity checks. Training uses v_old then teacher_velocity_from_old."""
        with self.torch.no_grad():
            v_old = self.predict_guided(latents, step, plain, "old", grad=False)
            v_plain = self.predict_guided(latents, step, plain, "base", grad=False)
            v_struct = self.predict_guided(latents, step, structured, "base", grad=False)
        proposal = propose_teacher(v_old, v_struct, v_plain, gamma=self.gamma)
        return proposal

    def finish_tail(self, latents, prompt: str, start: int, mode: LoraMode = "old", keep_steps=()):
        return self.run_ode_range(latents, prompt, start, self.steps, mode, keep_steps=keep_steps)

    def finish_and_decode(self, latents, prompt: str, start: int, mode: LoraMode = "old", keep_steps=()):
        latents, keep = self.finish_tail(latents, prompt, start, mode, keep_steps=keep_steps)
        images, _timing = self.decode_batch([latents])
        return images[0], keep

    def _decode_one(self, latents):
        torch = self.torch
        if latents.device.type != "cuda":
            latents = latents.to(self.device)
        latent_ids = self._latent_ids
        latent_height = 2 * (self.height // (self.pipe.vae_scale_factor * 2))
        latent_width = 2 * (self.width // (self.pipe.vae_scale_factor * 2))
        unpacked = self.pipe._unpack_latents_with_ids(
            latents, latent_ids, latent_height // 2, latent_width // 2
        )
        mean = self.pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(unpacked.device, unpacked.dtype)
        std = torch.sqrt(
            self.pipe.vae.bn.running_var.view(1, -1, 1, 1) + self.pipe.vae.config.batch_norm_eps
        ).to(unpacked.device, unpacked.dtype)
        unpacked = self.pipe._unpatchify_latents(unpacked * std + mean)
        image = self.pipe.vae.decode(unpacked, return_dict=False)[0].detach()
        return self.pipe.image_processor.postprocess(image, output_type="pil")[0]

    def decode(self, latents):
        images, _timing = self.decode_batch([latents])
        return images[0]

    def decode_batch(self, latents_list: list):
        """Decode many latents with at most one Transformer CPU/GPU round-trip."""
        import time

        torch = self.torch
        timing = {"offload_sec": 0.0, "decode_sec": 0.0}
        if not latents_list:
            return [], timing

        def _run():
            with torch.inference_mode():
                return [self._decode_one(latents) for latents in latents_list]

        if not self.decode_offload:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            images = _run()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            timing["decode_sec"] = time.perf_counter() - t0
            return images, timing

        transformer = self.pipe.transformer
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_off = time.perf_counter()
        transformer.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_decode0 = time.perf_counter()
        timing["offload_sec"] += t_decode0 - t_off
        try:
            images = _run()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            timing["decode_sec"] = time.perf_counter() - t_decode0
            return images, timing
        finally:
            t_on = time.perf_counter()
            transformer.to("cuda")
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            timing["offload_sec"] += time.perf_counter() - t_on

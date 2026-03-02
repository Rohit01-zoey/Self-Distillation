# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Ground-truth prefix distillation: the student sees prompt + first L tokens of the
golden completion and generates the rest; the teacher evaluates the full completion
as in DistilTrainer. L decays from an initial to a final value over steps/epochs.
"""

from typing import Any, Optional, Union

import torch
from accelerate.utils import gather_object
from trl.trainer.utils import is_conversational, maybe_apply_chat_template, nanmax, nanmin, pad
from distil_trainer import DistilTrainer
from distil_config import DistilConfig


def _get_gt_prefix_length(
    initial: int,
    final: int,
    step: int,
    max_steps: int,
    epoch: float,
    num_epochs: float,
    schedule: str,
) -> int:
    """Compute current prefix length L. L decays from initial to final."""
    if schedule == "linear_step":
        if max_steps is None or max_steps <= 0 or step is None:
            progress = 0.0
        else:
            progress = min(1.0, step / max_steps)
    elif schedule == "linear_epoch":
        if num_epochs is None or num_epochs <= 0 or epoch is None:
            progress = 0.0
        else:
            progress = min(1.0, epoch / num_epochs)
    else:
        progress = 0.0
    L = initial + (final - initial) * progress
    return max(final, min(initial, int(round(L))))


class GtPrefixDistilTrainer(DistilTrainer):
    """
    DistilTrainer with ground-truth prefix: the student is given prompt + first L
    tokens of the golden completion and completes the rest; the teacher evaluates
    the full sequence (prefix + student suffix) as usual. L decays with steps/epochs.

    Dataset must include a `golden_completion` column (str): the assistant's
    ground-truth response text (will be tokenized to obtain the prefix).

    Config: set `gt_prefix_initial_length` (e.g. 64), `gt_prefix_final_length`
    (e.g. 0), and `gt_prefix_schedule` ("linear_step" or "linear_epoch").
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._gt_prefix_initial = getattr(self.args, "gt_prefix_initial_length", None)
        self._gt_prefix_final = getattr(self.args, "gt_prefix_final_length", 0)
        self._gt_prefix_schedule = getattr(self.args, "gt_prefix_schedule", "linear_step")

    def _set_signature_columns_if_needed(self):
        super()._set_signature_columns_if_needed()
        if self._gt_prefix_initial is not None and hasattr(self, "_signature_columns"):
            if "golden_completion" not in self._signature_columns:
                self._signature_columns = list(self._signature_columns) + ["golden_completion"]

    def _get_gt_prefix_length(self) -> int:
        """Current prefix length L from schedule."""
        if self._gt_prefix_initial is None:
            return 0
        step = self.state.global_step if self.state is not None else 0
        max_steps = getattr(self.state, "max_steps", None) or 0
        epoch = self.state.epoch if self.state is not None else 0.0
        num_epochs = getattr(self.args, "num_train_epochs", 1.0) or 1.0
        return _get_gt_prefix_length(
            self._gt_prefix_initial,
            self._gt_prefix_final,
            step,
            max_steps,
            epoch,
            num_epochs,
            self._gt_prefix_schedule,
        )

    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        L = self._get_gt_prefix_length()

        # No GT prefix: use standard generation
        if L <= 0 or self._gt_prefix_initial is None:
            return super()._generate_and_score_completions(inputs)

        prompts = [x["prompt"] for x in inputs]
        teacher_prompts = [x["teacher_prompt"] for x in inputs]
        if "golden_completion" not in inputs[0]:
            raise KeyError(
                "GtPrefixDistilTrainer requires dataset column 'golden_completion' (str) when "
                "gt_prefix_initial_length is set."
            )
        golden_completions = [x["golden_completion"] for x in inputs]

        # Tokenize golden completions and take first L tokens (leave at least 1 for student to generate)
        tokenizer = self.processing_class
        gt_prefix_ids_list = []
        generation_prompts = []
        for i, (prompt, golden_text) in enumerate(zip(prompts, golden_completions)):
            if isinstance(golden_text, list):
                golden_text = "\n".join(golden_text)
            gt_ids = tokenizer.encode(golden_text, add_special_tokens=False)
            prefix_len = min(L, max(0, len(gt_ids) - 1))  # at least 1 token to generate
            gt_prefix_ids = gt_ids[:prefix_len]
            gt_prefix_ids_list.append(gt_prefix_ids)
            gt_prefix_text = tokenizer.decode(gt_prefix_ids, skip_special_tokens=False)
            prompt_text = maybe_apply_chat_template({"prompt": prompt}, tokenizer)["prompt"]
            generation_prompts.append(prompt_text + gt_prefix_text)

        if "images" in inputs[0]:
            images = [x.get("images") for x in inputs]
        elif "image" in inputs[0]:
            images = [[x.get("image")] if x.get("image") is not None else None for x in inputs]
        else:
            images = None
        if images is not None and all(img_list == [] for img_list in images):
            images = None

        # Limit suffix length so prefix + suffix <= max_completion_length
        max_suffix = max(1, self.max_completion_length - L)
        old_max_completion = self.max_completion_length
        old_max_new_tokens = getattr(self.generation_config, "max_new_tokens", None)
        self.max_completion_length = max_suffix
        if self.generation_config is not None:
            self.generation_config.max_new_tokens = max_suffix
        try:
            (
                _,
                suffix_ids_list,
                total_suffix_tokens,
                sampling_per_token_logps_list,
                forward_kwargs,
            ) = self._generate(generation_prompts, images)
            # Importance sampling logprobs are for suffix only; full completion = prefix + suffix, so we skip
            # importance sampling correction in the GT-prefix path to avoid shape mismatch.
            sampling_per_token_logps_list = None
        finally:
            self.max_completion_length = old_max_completion
            if self.generation_config is not None and old_max_new_tokens is not None:
                self.generation_config.max_new_tokens = old_max_new_tokens

        # Full completion = GT prefix + student suffix
        full_completion_ids_list = [
            gt_prefix_ids_list[i] + suffix_ids_list[i]
            for i in range(len(gt_prefix_ids_list))
        ]
        completion_ids_list = full_completion_ids_list
        num_items_in_batch = sum(len(c) for c in full_completion_ids_list)

        # Build mask for loss: 1 only on student-generated tokens (suffix), 0 on GT prefix and padding.
        # Teacher (and loss) will only evaluate the student-generated part.
        prefix_lengths = [len(gt_prefix_ids_list[i]) for i in range(len(gt_prefix_ids_list))]
        completion_lengths = [len(full_completion_ids_list[i]) for i in range(len(full_completion_ids_list))]

        # Rest of the pipeline: same as DistilTrainer using our completion_ids_list
        mode = "train" if self.model.training else "eval"
        prompts_text = [
            maybe_apply_chat_template({"prompt": p}, self.processing_class)["prompt"] for p in prompts
        ]
        if self.use_vllm:
            self.processing_class.truncation_side = "left"
        student_inputs = self.processing_class(
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            max_length=self.max_prompt_length,
            truncation=True,
            add_special_tokens=False,
        )
        student_inputs = super()._prepare_inputs(student_inputs)
        student_prompt_ids, student_prompt_mask = student_inputs["input_ids"], student_inputs["attention_mask"]
        prompt_ids_list = [p[m].tolist() for p, m in zip(student_prompt_ids, student_prompt_mask.bool())]

        teacher_prompts_text = [
            maybe_apply_chat_template({"prompt": p}, self.processing_class)["prompt"] for p in teacher_prompts
        ]
        teacher_inputs = self.processing_class(
            text=teacher_prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            max_length=self.max_prompt_length,
            truncation=True,
            add_special_tokens=False,
        )
        teacher_inputs = super()._prepare_inputs(teacher_inputs)
        if self.use_vllm:
            self.processing_class.truncation_side = "right"
        teacher_prompt_ids, teacher_prompt_mask = teacher_inputs["input_ids"], teacher_inputs["attention_mask"]
        teacher_prompt_ids_list = [p[m].tolist() for p, m in zip(teacher_prompt_ids, teacher_prompt_mask.bool())]

        prompt_ids = [torch.tensor(ids, device=device) for ids in prompt_ids_list]
        prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in prompt_ids]
        prompt_ids = pad(prompt_ids, padding_value=self.pad_token_id, padding_side="left")
        prompt_mask = pad(prompt_mask, padding_value=0, padding_side="left")
        teacher_prompt_ids = [torch.tensor(ids, device=device) for ids in teacher_prompt_ids_list]
        teacher_prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in teacher_prompt_ids]
        teacher_prompt_ids = pad(teacher_prompt_ids, padding_value=self.pad_token_id, padding_side="left")
        teacher_prompt_mask = pad(teacher_prompt_mask, padding_value=0, padding_side="left")
        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids_list]
        completion_mask = [torch.ones_like(ids, dtype=torch.long) for ids in completion_ids]
        completion_ids = pad(completion_ids, padding_value=self.pad_token_id, padding_side="right")
        completion_mask = pad(completion_mask, padding_value=0, padding_side="right")

        # gt_prefix_loss_mask: 1 on student-generated (suffix) positions, 0 on GT prefix and padding
        batch_size = completion_ids.size(0)
        seq_len = completion_ids.size(1)
        gt_prefix_loss_mask = torch.zeros(batch_size, seq_len, device=device, dtype=completion_mask.dtype)
        for i in range(batch_size):
            pl, cl = prefix_lengths[i], completion_lengths[i]
            if cl > pl:
                gt_prefix_loss_mask[i, pl:cl] = 1

        if sampling_per_token_logps_list is not None:
            sampling_per_token_logps = [
                torch.tensor(logps, device=device) for logps in sampling_per_token_logps_list
            ]
            sampling_per_token_logps = pad(sampling_per_token_logps, padding_value=0.0, padding_side="right")
        else:
            sampling_per_token_logps = None

        if self.mask_truncated_completions:
            eos_and_pad = [self.eos_token_id, self.pad_token_id]
            is_truncated = torch.tensor(
                [ids[-1] not in eos_and_pad for ids in completion_ids_list], device=device
            )
            completion_mask = completion_mask * (~is_truncated).unsqueeze(1).int()

        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        teacher_prompt_completion_ids = torch.cat([teacher_prompt_ids, completion_ids], dim=1)
        teacher_attention_mask = torch.cat([teacher_prompt_mask, completion_mask], dim=1)
        if "token_type_ids" in forward_kwargs:
            token_type_ids = forward_kwargs["token_type_ids"]
            forward_kwargs["token_type_ids"] = torch.cat(
                [token_type_ids, token_type_ids.new_zeros(completion_ids.shape)], dim=1
            )

        logits_to_keep = completion_ids.size(1)
        batch_size = (
            self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size
        )
        num_images = [len(img_list) for img_list in images] if images is not None else None

        with torch.no_grad():
            generate_every = self.args.steps_per_generation * self.num_iterations
            if not self.generate_from_teacher and (
                self.args.gradient_accumulation_steps % generate_every != 0
                or (self.use_vllm and self.vllm_importance_sampling_correction)
            ):
                old_per_token_logps, _, _ = self._get_per_token_logps_and_entropies(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size,
                    num_images=num_images,
                    compute_all_logps=False,
                    **forward_kwargs,
                )
            else:
                old_per_token_logps = None

            if self.use_vllm and self.vllm_importance_sampling_correction and not self.generate_from_teacher:
                importance_sampling_ratio = torch.exp(old_per_token_logps - sampling_per_token_logps)
                importance_sampling_ratio = torch.clamp(
                    importance_sampling_ratio, max=self.vllm_importance_sampling_cap
                )
            else:
                importance_sampling_ratio = None

            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_per_token_logps, _, _ = self._get_per_token_logps_and_entropies(
                        self.ref_model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size=batch_size,
                        num_images=num_images,
                        compute_all_logps=False,
                        **forward_kwargs,
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps, _, _ = self._get_per_token_logps_and_entropies(
                            self.model,
                            prompt_completion_ids,
                            attention_mask,
                            logits_to_keep,
                            batch_size=batch_size,
                            num_images=num_images,
                            compute_all_logps=False,
                            **forward_kwargs,
                        )
            else:
                ref_per_token_logps = None

        prompts_text = self.processing_class.batch_decode(prompt_ids, skip_special_tokens=True)
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt[-1]["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        rewards = torch.zeros_like(completion_ids, dtype=torch.float32)
        advantages = rewards
        all_process_advantages = advantages.clone()

        self._logs["prompt"].extend(gather_object(prompts_text))
        self._logs["completion"].extend(gather_object(completions_text))
        self._logs["rewards"]["main"].extend(gather_object(rewards.mean(dim=-1).tolist()))
        self._logs["advantages"].extend(gather_object(all_process_advantages.mean(dim=-1).tolist()))
        reward_to_log = rewards.clone()
        reward_to_log = reward_to_log[completion_mask.bool()]
        mean_reward = torch.mean(reward_to_log) if reward_to_log.numel() > 0 else torch.tensor(0.0, device=device)
        self._metrics[mode]["rewards"].append(self.accelerator.gather(mean_reward).mean().item())

        if images is not None:
            self._logs["images"].extend(gather_object(images))

        if importance_sampling_ratio is not None:
            delta = torch.abs(old_per_token_logps - sampling_per_token_logps)
            delta = delta[completion_mask.bool()]
            mean_delta = torch.mean(delta) if delta.numel() > 0 else torch.tensor(0.0, device=device)
            max_delta = torch.max(delta) if delta.numel() > 0 else torch.tensor(0.0, device=device)
            self._metrics[mode]["sampling/sampling_logp_difference/mean"].append(
                self.accelerator.gather(mean_delta).mean().item()
            )
            self._metrics[mode]["sampling/sampling_logp_difference/max"].append(
                self.accelerator.gather(max_delta).max().item()
            )
            flat_is_ratio = importance_sampling_ratio[completion_mask.bool()]
            min_is = torch.min(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
            mean_is = torch.mean(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
            max_is = torch.max(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
            self._metrics[mode]["sampling/importance_sampling_ratio/min"].append(
                nanmin(self.accelerator.gather(min_is)).item()
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/mean"].append(
                self.accelerator.gather(mean_is).nanmean().item()
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/max"].append(
                nanmax(self.accelerator.gather(max_is)).item()
            )

        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "teacher_prompt_ids": teacher_prompt_ids,
            "teacher_prompt_mask": teacher_prompt_mask,
            "advantages": advantages,
            "num_items_in_batch": num_items_in_batch,
            "gt_prefix_loss_mask": gt_prefix_loss_mask,
        }
        if old_per_token_logps is not None:
            output["old_per_token_logps"] = old_per_token_logps
        if importance_sampling_ratio is not None:
            output["importance_sampling_ratio"] = importance_sampling_ratio
        if ref_per_token_logps is not None:
            output["ref_per_token_logps"] = ref_per_token_logps
        for key in ("pixel_values", "image_grid_thw", "pixel_attention_mask", "image_sizes"):
            if key in forward_kwargs:
                output[key] = forward_kwargs[key]

        return output

    def _compute_loss(self, model, inputs):
        """Same as DistilTrainer but mask out GT prefix tokens from loss (only evaluate student-generated part)."""
        completion_mask = inputs["completion_mask"]
        # Apply GT prefix mask when present: loss only on student-generated (suffix) tokens
        gt_prefix_loss_mask = inputs.get("gt_prefix_loss_mask")
        if gt_prefix_loss_mask is not None:
            loss_completion_mask = completion_mask * gt_prefix_loss_mask
        else:
            loss_completion_mask = completion_mask

        if self.num_loss_tokens_to_skip > 0:
            batch_size, seq_len = completion_mask.shape
            token_positions = (
                torch.arange(seq_len, device=completion_mask.device).unsqueeze(0).expand(batch_size, -1)
            )
            skip_mask = (token_positions >= self.num_loss_tokens_to_skip).int()
            loss_completion_mask = loss_completion_mask * skip_mask

        # Build inputs for base logic with our loss_completion_mask; then run standard forward + KL
        inputs_with_loss_mask = {**inputs, "loss_completion_mask": loss_completion_mask}
        return self._compute_loss_with_mask(model, inputs_with_loss_mask)

    def _compute_loss_with_mask(self, model, inputs):
        """DistilTrainer._compute_loss logic with a precomputed loss_completion_mask."""
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        teacher_prompt_ids, teacher_prompt_mask = inputs["teacher_prompt_ids"], inputs["teacher_prompt_mask"]
        loss_completion_mask = inputs["loss_completion_mask"]

        from torch.nn.functional import kl_div

        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        teacher_input_ids = torch.cat([teacher_prompt_ids, completion_ids], dim=1)
        teacher_attention_mask = torch.cat([teacher_prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        per_token_logps, all_logps, entropies = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            compute_entropy=True,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            num_images=inputs.get("num_images"),
            pixel_attention_mask=inputs.get("pixel_attention_mask"),
            image_sizes=inputs.get("image_sizes"),
            token_type_ids=inputs.get("token_type_ids"),
        )

        with torch.no_grad():
            teacher_per_token_logps, teacher_all_logps, teacher_entropies = self._get_per_token_logps_and_entropies(
                self.ref_model,
                teacher_input_ids,
                teacher_attention_mask,
                logits_to_keep,
                compute_entropy=True,
                pixel_values=inputs.get("pixel_values"),
                image_grid_thw=inputs.get("image_grid_thw"),
                num_images=inputs.get("num_images"),
                pixel_attention_mask=inputs.get("pixel_attention_mask"),
                image_sizes=inputs.get("image_sizes"),
                token_type_ids=inputs.get("token_type_ids"),
            )

        if self.top_entropy_quantile < 1.0:
            entropy_mask = self.get_high_entropy_mask(
                entropies, loss_completion_mask, 1 - self.top_entropy_quantile
            )
        else:
            entropy_mask = None

        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps)
                - 1
            )

        if self.alpha == 0:
            kl_loss = kl_div(all_logps, teacher_all_logps, reduction="none", log_target=True)
        elif self.alpha == 1:
            kl_loss = kl_div(teacher_all_logps, all_logps, reduction="none", log_target=True)
        else:
            alpha = torch.tensor(self.alpha, dtype=all_logps.dtype)
            mixture_log_probs = torch.logsumexp(
                torch.stack(
                    [all_logps + torch.log(1 - alpha), teacher_all_logps + torch.log(alpha)]
                ),
                dim=0,
            )
            kl_teacher = kl_div(mixture_log_probs, teacher_all_logps, reduction="none", log_target=True)
            kl_student = kl_div(mixture_log_probs, all_logps, reduction="none", log_target=True)
            kl_loss = alpha * kl_teacher + (1 - alpha) * kl_student
        per_token_loss = kl_loss.sum(-1)

        if (
            self.use_vllm
            and self.vllm_importance_sampling_correction
            and not self.generate_from_teacher
            and inputs.get("importance_sampling_ratio") is not None
        ):
            ratio = inputs["importance_sampling_ratio"]
            importance_weights = (
                (ratio * loss_completion_mask).sum(-1) / loss_completion_mask.sum(-1).clamp(min=1.0)
            )
            importance_weights = importance_weights.unsqueeze(-1)
            per_token_loss = per_token_loss * importance_weights

        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask

        loss = (
            (per_token_loss * loss_completion_mask).sum(-1)
            / loss_completion_mask.sum(-1).clamp(min=1.0)
        ).mean()
        loss = loss / self.current_gradient_accumulation_steps

        mode = "train" if self.model.training else "eval"
        with torch.no_grad():
            kl_approx = (
                (per_token_logps - teacher_per_token_logps)
                + torch.exp(teacher_per_token_logps - per_token_logps)
                - 1
            )
            kl_approx_mean = (kl_approx * loss_completion_mask).sum() / loss_completion_mask.sum()
        self._metrics[mode]["kl_approx"].append(self.accelerator.gather(kl_approx_mean).nanmean().item())

        loss_completion_token_count = loss_completion_mask.sum().clamp(min=1.0)

        def masked_batch_mean(x):
            if x.shape[1] == 1:
                return x.mean()
            else:
                return (x * loss_completion_mask).sum() / loss_completion_token_count

        if self.beta != 0.0:
            mean_kl = masked_batch_mean(per_token_kl)
            self._metrics[mode]["kl_to_base_model"].append(
                self.accelerator.gather(mean_kl).nanmean().item()
            )
        mean_entropy = masked_batch_mean(entropies)
        self._metrics[mode]["entropy"].append(self.accelerator.gather(mean_entropy).nanmean().item())

        return loss

from typing import List, Optional, Tuple

import torch
import transformers

from self_speculation.generator_base import (
    GenerationConfig,
    GenerationStrategy,
    GenerationStrategyResult,
)
from self_speculation.llama_model_utils import (
    crop_past_key_values,
    decode_next_token,
    forward,
)


def max_fn(x, eps=1e-6):
    x_max = torch.where(x > 0, x, 0)
    return x_max / (torch.sum(x_max) + eps)


def fly_loosely_match(verified, win_len):
    original_verified = verified.clone()
    pattern = torch.ones(win_len, dtype=torch.bool, device=verified.device)
    pattern[0] = False
    unfold = verified.unfold(1, win_len, 1)
    matched = torch.all(unfold == pattern, dim=-1)
    fly_mask = torch.zeros_like(verified)
    fly_mask[:, :matched.shape[1]] = matched
    verified = verified | fly_mask
    verified[:, -win_len:] = verified[:, -win_len:] & original_verified[:, -win_len:]
    return verified


class FLySpeculativeGenerationStrategy(GenerationStrategy):
    def __init__(self):
        super().__init__()
        self.draft_model = None
        self.enable_fly = False
        self.fly_win_len = 6

    def generate_token_ids(
        self,
        model: transformers.LlamaForCausalLM,
        input_ids: List[int],
        eos_token_ids: List[int],
        generation_config: GenerationConfig,
        logits_processors: Optional[transformers.generation.logits_process.LogitsProcessorList] = None,
        stopping_criteria: Optional[transformers.StoppingCriteriaList] = None,
        streamer: Optional[transformers.TextStreamer] = None,
    ) -> GenerationStrategyResult:
        self.enable_fly = generation_config.enable_fly
        self.fly_win_len = generation_config.fly_win_len

        draft_past_key_values = None
        target_past_key_values = None
        input_ids_list = input_ids
        input_ids: torch.Tensor = torch.tensor([input_ids_list]).to(model.device)
        output_ids: List[int] = []

        total_draft_matches = 0
        total_generations = 0
        while len(output_ids) < generation_config.max_steps:
            (
                input_ids,
                output_ids,
                draft_past_key_values,
                target_past_key_values,
                number_of_matches,
                num_speculations,
            ) = self.single_step_speculation(
                target_model=model,
                draft_model=self.draft_model,
                input_ids_list=input_ids_list,
                input_ids=input_ids,
                output_ids=output_ids,
                num_speculations=min(
                    generation_config.num_speculations,
                    generation_config.max_steps - len(output_ids) - 1,
                ),
                draft_past_key_values=draft_past_key_values,
                target_past_key_values=target_past_key_values,
                eos_token_ids=eos_token_ids,
                sample=generation_config.sample,
                temperature=generation_config.temperature,
                top_k=generation_config.top_k,
                top_p=generation_config.top_p,
                logits_processors=logits_processors,
                stopping_criteria=stopping_criteria,
            )
            total_draft_matches += number_of_matches
            total_generations += num_speculations

            eos_found = False
            for eos_token_id in eos_token_ids:
                if eos_token_id in output_ids:
                    output_ids = output_ids[: output_ids.index(eos_token_id)]
                    eos_found = True
                    break
            if eos_found:
                break
            if stopping_criteria:
                if torch.all(stopping_criteria(input_ids, scores=None)):
                    break

        return GenerationStrategyResult(
            predicted_tokens=output_ids,
            acceptance_rate=total_draft_matches / total_generations if total_generations > 0 else 0,
        )

    def single_step_speculation(
        self,
        target_model: transformers.LlamaForCausalLM,
        draft_model: transformers.LlamaForCausalLM,
        input_ids: torch.Tensor,
        input_ids_list: List[int],
        output_ids: List[int],
        num_speculations: int,
        draft_past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
        target_past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
        eos_token_ids: List[int],
        sample: Optional[bool] = False,
        temperature: Optional[float] = 0.7,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.95,
        logits_processors: Optional[transformers.generation.logits_process.LogitsProcessorList] = None,
        stopping_criteria: Optional[transformers.StoppingCriteriaList] = None,
    ):
        prompt_length: int = input_ids.size(1)
        draft_input_ids = input_ids.clone().to(draft_model.device)
        draft_output_ids: List[int] = []
        if sample:
            draft_probabilities: List[torch.Tensor] = []

        # --- Draft phase: autoregressive with draft model ---
        for _ in range(num_speculations):
            draft_result = forward(draft_model, draft_input_ids, draft_past_key_values)
            draft_past_key_values = draft_result.past_key_values
            draft_logits = draft_result.logits
            if logits_processors:
                draft_logits = logits_processors(draft_input_ids, draft_logits)

            draft_next_token, draft_next_prob = decode_next_token(
                logits=draft_logits, token_idx=-1,
                sample=sample, temperature=temperature, top_k=top_k, top_p=top_p,
            )
            draft_next_token = draft_next_token.item()
            draft_output_ids.append(draft_next_token)
            if sample:
                draft_probabilities.append(draft_next_prob)
            draft_input_ids = torch.tensor([[draft_next_token]]).to(draft_model.device)
            if draft_next_token in eos_token_ids:
                break

        # --- Verify phase: parallel forward with target model ---
        draft_output_ids = torch.tensor(draft_output_ids).unsqueeze(0).to(target_model.device)
        prefill_token_ids = torch.cat(
            [input_ids.to(target_model.device), draft_output_ids], dim=-1,
        )

        verify_result = forward(target_model, prefill_token_ids.int(), target_past_key_values)
        logits = verify_result.logits
        if logits_processors:
            logits = logits_processors(prefill_token_ids, logits)
        target_past_key_values = verify_result.past_key_values

        verification_logits = logits[:, prompt_length - 1:, :]
        verified_tokens, verified_probabilities = decode_next_token(
            logits=verification_logits,
            sample=sample, temperature=temperature, top_k=top_k, top_p=top_p,
        )
        verified_tokens = verified_tokens.to(prefill_token_ids)
        verified = draft_output_ids[:, :] == verified_tokens[:, :-1]

        # --- FLy loosely matching ---
        if self.enable_fly and not sample and verified.shape[1] >= self.fly_win_len:
            verified = fly_loosely_match(verified, self.fly_win_len)

        # --- Accept / reject ---
        if not sample:
            number_of_matches = ((~verified).cumsum(dim=-1) < 1).sum().item()
        else:
            number_of_matches = 0
            rand = torch.rand_like(draft_output_ids, dtype=torch.float)
            for i in range(draft_output_ids.numel()):
                if rand[0, i] < min(1, verified_probabilities[i, draft_output_ids[0, i]].item() / draft_probabilities[i][0, draft_output_ids[0, i]].item()):
                    number_of_matches += 1
                else:
                    verified_tokens[0][number_of_matches] = torch.multinomial(
                        max_fn(verified_probabilities[i, :] - draft_probabilities[i]),
                        num_samples=1,
                    ).item()
                    break

        input_ids = verified_tokens[:, number_of_matches: number_of_matches + 1]
        output_ids.extend(draft_output_ids[0, :number_of_matches].tolist())
        output_ids.extend(verified_tokens[0][number_of_matches: number_of_matches + 1].tolist())

        accepted_length = len(input_ids_list) + len(output_ids) - 1
        draft_past_key_values = crop_past_key_values(draft_past_key_values, accepted_length)
        target_past_key_values = crop_past_key_values(target_past_key_values, accepted_length)

        return (
            input_ids,
            output_ids,
            draft_past_key_values,
            target_past_key_values,
            number_of_matches,
            draft_output_ids.numel(),
        )

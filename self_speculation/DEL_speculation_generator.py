# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import List, Optional, Tuple

import colorama
import torch
import itertools
import matplotlib.pyplot as plt
import os

import transformers
from self_speculation.generator_base import (
    GenerationConfig,
    GenerationStrategy,
    GenerationStrategyResult,
)
from self_speculation.speculative_streamer import SpeculativeTextStreamer
from self_speculation.llama_model_utils import (
    crop_past_key_values,
    decode_next_token,
    forward_early_DEL,
    forward_remainder_DEL,
)
from self_speculation.DEL import DEL



def max_fn(x, eps=1e-6):
    x_max = torch.where(x > 0, x, 0)
    return x_max / (torch.sum(x_max) + eps)


class DELSpeculativeGenerationStrategy(GenerationStrategy):
    def __init__(self):
        super().__init__()
        self.DEL = None
    
    def generate_token_ids(
        self,
        model: torch.nn.Module,
        input_ids: List[int],
        eos_token_ids: List[int],
        generation_config: GenerationConfig,
        logits_processors: Optional[transformers.generation.logits_process.LogitsProcessorList] = None,
        stopping_criteria: Optional[transformers.StoppingCriteriaList] = None,
        streamer: Optional[transformers.TextStreamer] = None,
    ) -> GenerationStrategyResult:
        past_key_values = None
        input_ids_list = input_ids
        input_ids: torch.Tensor = torch.tensor([input_ids_list]).to(model.device)
        output_ids: List[int] = []

        # DEL Module Definition
        self.DEL = DEL(model, gamma_max=18, omega=0.95)
        self.enable_fly = generation_config.enable_fly
        self.fly_win_len = generation_config.fly_win_len
       
        # DEL with fixed exit_layer
        # self.DEL = DEL(model, eligible_exit_layers=list(range(generation_config.exit_layer-1, generation_config.exit_layer)), gamma_max=18, omega=0.95) 
        
        # DEL with limited exit_layer range
        # self.DEL = DEL(model, eligible_exit_layers=list(range(3, len(model.model.layers)-1)), gamma_max=18, omega = 0.95) 
        

        calls: int = 0
        total_draft_matches = 0
        total_generations = 0
        total_layers = 0
        total_tokens = 0
        while len(output_ids) < generation_config.max_steps:
            current_exit_layer = self.DEL.current_exit_layer
            (
                input_ids,
                output_ids,
                past_key_values,
                number_of_matches,
                num_speculations,
            ) = self.single_step_speculation(
                model=model,
                input_ids_list=input_ids_list,
                input_ids=input_ids,
                output_ids=output_ids,
                num_speculations=min(
                    self.DEL.current_gamma,
                    generation_config.max_steps - len(output_ids) - 1,
                ),
                past_key_values=past_key_values,
                exit_layer=self.DEL.current_exit_layer,
                eos_token_ids=eos_token_ids,
                calls=calls,
                sample=generation_config.sample,
                temperature=generation_config.temperature,
                top_k=generation_config.top_k,
                top_p=generation_config.top_p,
                logits_processors=logits_processors,
                stopping_criteria=stopping_criteria,
                streamer=streamer,
            )
            calls += 1
            total_draft_matches += number_of_matches
            total_generations += num_speculations
            total_tokens += (number_of_matches+1)
            total_layers += current_exit_layer * num_speculations + len(model.model.layers)
            
            eos_found = False
            for eos_token_id in eos_token_ids:
                if eos_token_id in output_ids:
                    # break out of loop when we get an EOS token
                    # remove the EOS token id
                    output_ids = output_ids[: output_ids.index(eos_token_id)]
                    eos_found = True
                    break
            if eos_found:
                break
            if stopping_criteria:
                # TODO: when implementing batch size > 1, stop each sample separately?
                if torch.all(stopping_criteria(input_ids, scores=None)):
                    break
        
        return GenerationStrategyResult(
            predicted_tokens=output_ids,
            acceptance_rate=total_draft_matches / total_generations if total_generations > 0 else 0,
            tokens_per_layer=total_tokens / total_layers if total_layers > 0 else 0,
        )

    # TODO: remove calls, input_ids_list, rely on generation config
    def single_step_speculation(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        input_ids_list: List[int],
        output_ids: List[int],
        num_speculations: int,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
        eos_token_ids: List[int],
        calls: int,
        exit_layer: int,
        sample: Optional[bool] = False,
        temperature: Optional[float] = 0.7,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.95,
        logits_processors: Optional[transformers.generation.logits_process.LogitsProcessorList] = None,
        stopping_criteria: Optional[transformers.StoppingCriteriaList] = None,
        streamer: Optional[transformers.TextStreamer] = None
    ):
        prompt_length: int = input_ids.size(1)
        draft_input_ids = input_ids.clone()
        draft_output_ids: List[int] = []
        if sample:
            draft_probabilities: List[torch.Tensor] = []
        exit_query_cache = None

        d_max = 1 if self.DEL.is_prefill_stage else 18
        for di in range(d_max):
            draft_result = forward_early_DEL(
                model,
                draft_input_ids,
                past_key_values,
                exit_layer,
                exit_query_cache,
                self.DEL,
            )
            past_key_values = draft_result.past_key_values
            exit_query_cache = draft_result.exit_query_cache
            draft_logits = draft_result.logits
            if logits_processors:
                draft_logits = logits_processors(draft_input_ids, draft_logits)

            draft_next_token, draft_next_prob = decode_next_token(logits=draft_logits, token_idx=-1, sample=sample, temperature=temperature, top_k=top_k, top_p=top_p)
            draft_next_token = draft_next_token.item()
            draft_output_ids.append(draft_next_token)
            if sample:
                draft_probabilities.append(draft_next_prob)
            draft_input_ids = torch.tensor([[draft_next_token]]).to(draft_input_ids)
            if draft_next_token in eos_token_ids:
                # break out of loop when we get an EOS token
                break

            # Get DEL threshold τₗ for this exit layer
            draft_token_confidence_score = draft_next_prob[0, draft_next_token].item()
            avg_confidence_stats = self.DEL.average_confidence[exit_layer - 1]  # [τ_match, τ_mismatch]
            tau_threshold_for_exit_layer = ((avg_confidence_stats[0] + avg_confidence_stats[1]) / 2).item()

            if (not self.DEL.is_prefill_stage) and draft_token_confidence_score < tau_threshold_for_exit_layer:
                break
                
        # input_ids (1 x T_p) and draft_output_ids (1 x T_d) are concatenated together to make
        # 1 x (T_d  + T_p)
        draft_output_ids = torch.tensor(draft_output_ids).unsqueeze(0).to(input_ids)
        prefill_token_ids = torch.cat(
            [input_ids, draft_output_ids],
            dim=-1,
        )

        if streamer:
            if isinstance(streamer, SpeculativeTextStreamer):
                print(colorama.Fore.LIGHTMAGENTA_EX, end="")
                streamer.put(draft_output_ids, is_draft=True)

        # logits: 1 x (T_d  + T_p) x V
        verify_results = forward_remainder_DEL(
            model,
            prefill_token_ids.int(),
            past_key_values,
            exit_layer,
            exit_query_cache,
            self.DEL,
        )
        logits, del_tokens = self.DEL.run(exit_layer, prompt_length-1, sample)

        if logits_processors:
            logits = logits_processors(prefill_token_ids, logits)
        past_key_values = verify_results.past_key_values
        # only select the logits relevant to what the draft has outputted.
        # verification_logits: 1 x T_d x V
        verification_logits = logits[:, prompt_length - 1 :, :]

        # verified_tokens: 1 x (T_d)
        # There is a predicted token for every token in the draft output ids list, however note that the
        # first tokens (or first N tokens) are coming from the prompt
        if sample:
            verified_tokens, verified_probabilities = decode_next_token(logits=verification_logits, sample=sample, temperature=temperature, top_k=top_k, top_p=top_p)
        else:
            verified_tokens = del_tokens
            verified_probabilities = None

        # skip verification of the last token as it is a new token predicted from the main model
        verified_tokens = verified_tokens.to(prefill_token_ids)
        verified = draft_output_ids[:, :] == verified_tokens[:, :-1]

        # FLy: loosely accept draft tokens using a sliding window pattern
        if self.enable_fly and not sample and verified.shape[1] >= self.fly_win_len:
            original_verified = verified.clone()
            pattern = torch.ones(self.fly_win_len, dtype=torch.bool, device=verified.device)
            pattern[0] = False
            unfold = verified.unfold(1, self.fly_win_len, 1)
            matched = torch.all(unfold == pattern, dim=-1)
            fly_mask = torch.zeros_like(verified)
            fly_mask[:, :matched.shape[1]] = matched
            verified = verified | fly_mask
            verified[:, -self.fly_win_len:] = (
                verified[:, -self.fly_win_len:] & original_verified[:, -self.fly_win_len:]
            )

        # number of matches is the index of the number of tokens we are accepting from the draft
        if not sample:
            number_of_matches = ((~(verified)).cumsum(dim=-1) < 1).sum().item()
        else:
            number_of_matches = 0
            rand = torch.rand_like(draft_output_ids, dtype=torch.float)
            for i in range(draft_output_ids.numel()):
                if rand[0, i] < min(1, verified_probabilities[i, draft_output_ids[0, i]].item() / draft_probabilities[i][0, draft_output_ids[0, i]].item()):
                    number_of_matches += 1
                else:
                    verified_tokens[0][number_of_matches] = torch.multinomial(max_fn((verified_probabilities[i, :] - draft_probabilities[i])), num_samples=1).item()
                    break

        # accept the `number_of_matches` tokens from the draft with one more from the main model
        # since we re-use the same cachem the input id should only be the last accepted token TODO check this
        input_ids = verified_tokens[:, number_of_matches : number_of_matches + 1]
        output_ids.extend(draft_output_ids[0, : number_of_matches].tolist())
        output_ids.extend(verified_tokens[0][number_of_matches : number_of_matches + 1].tolist())

        if streamer:
            if isinstance(streamer, SpeculativeTextStreamer):
                streamer.delete(len(draft_output_ids[0, :]))
                print(colorama.Fore.GREEN, end="")
                streamer.put(draft_output_ids[0, : number_of_matches])
                print(colorama.Style.RESET_ALL, end="")
                streamer.put(verified_tokens[0][number_of_matches : number_of_matches + 1])
            else:
                streamer.put(torch.LongTensor(output_ids[len(output_ids)-number_of_matches-1:]))

        # we want the entire output sequence + input sequence
        past_key_values = crop_past_key_values(
            past_key_values, len(input_ids_list) + len(output_ids) - 1
        )
        
        return (
            input_ids,
            output_ids,
            past_key_values,
            number_of_matches,
            draft_output_ids.numel(),
        )

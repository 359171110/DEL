# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import json
import random
from dataclasses import dataclass
from os.path import split
from typing import Dict, List, Optional

from datasets import load_dataset
import pandas as pd

# for language modeling problems how long to use the prefix as
PREFIX_LENGTH: int = 100


@dataclass
class EvaluationExample:
    input: str
    output: str


class DatasetFormat:
    CHAT_FORMAT: str = "chat_format"
    CNN_DM_SUMMARIZATION: str = "cnn_dm_summarization"
    CNN_DM_LM: str = "cnn_dm_lm"
    XSUM_SUMMARIZATION: str = "xsum_summarization"
    HUMAN_EVAL: str = "human_eval"
    GSM8k: str = "gsm8k"
    AQUA_RAT: str = "aqua_rat"
    FINANCE_ALPACA: str = "finance_alpaca"
    WMT14_DE_EN: str = "wmt14_de_en"
    ALPACA: str = "alp"
    CUSTOM_JSONL: str = "custom_jsonl"


def LowercaseProcessingFunction(input: str) -> str:
    return input.lower()


# TODO: fix or remove TOPv2 benchmarking
def prepare_evaluation_examples_chat_format(data_path: str) -> List[EvaluationExample]:
    SINGLE_TURN_TEMPLATE: str = "\n[{role}]\n{message}\n[/{role}]"
    evaluation_data_points = []

    def stringify_conversation(conversation: List[Dict[str, str]]) -> str:
        return "".join(
            [
                SINGLE_TURN_TEMPLATE.format(role=x["role"], message=x["message"])
                for x in conversation
            ]
        )

    for line in open(data_path):
        json_line = json.loads(line)
        i: int = 0
        while i < len(json_line["data"]):
            if json_line["data"][i]["role"] == "PARSER":
                evaluation_data_points.append(
                    EvaluationExample(
                        input=stringify_conversation(json_line["data"][1:i])
                        + "\n[PARSER]\n",
                        output=stringify_conversation([json_line["data"][i]]),
                    )
                )
            i += 1
    return evaluation_data_points


def prepare_cnn_dm_lm_format() -> List[EvaluationExample]:
    evaluation_data_points = []
    for data_point in load_dataset("cnn_dailymail", "3.0.0")["test"]:
        words = data_point["article"].split()
        evaluation_data_points.append(
            EvaluationExample(
                input=" ".join(words[:PREFIX_LENGTH]),
                output=" ".join(words[PREFIX_LENGTH:]),
            )
        )
    return evaluation_data_points

def prepare_cnn_dm_summarization_format(n_shot: int = 0, seed: int = 42) -> List[EvaluationExample]:
    prompt_shots = ""
    if n_shot > 0:
        prompt_keys=["article", "highlights"]
        shots = load_dataset("cnn_dailymail", name="3.0.0", split="train").shuffle(seed=seed).select(range(n_shot))
        for i in range(n_shot):
            prompt = "Article: " + shots[i][prompt_keys[0]] + "\nSummary: " + shots[i][prompt_keys[1]].replace("\n", "") + "\n"
            prompt_shots += prompt
        prompt_shots += "\n"

    evaluation_data_points = []
    for data_point in load_dataset("cnn_dailymail", name="3.0.0", split="test"):
        article = data_point["article"]
        highlights = data_point["highlights"]
        evaluation_data_points.append(
            EvaluationExample(
                input=prompt_shots + f"Article: {article}\nSummary:",
                output=f" {highlights}",
            )
        )
    return evaluation_data_points

def prepare_xsum_summarization_format(n_shot: int = 0, seed: int = 42) -> List[EvaluationExample]:
    prompt_shots = ""
    if n_shot > 0:
        prompt_keys=["document", "summary"]
        shots = load_dataset("xsum", split="train").shuffle(seed=seed).select(range(n_shot))
        for i in range(n_shot):
            prompt = "Article: " + shots[i][prompt_keys[0]] + "\nSummary: " + shots[i][prompt_keys[1]].replace("\n", "") + "\n"
            prompt_shots += prompt
        prompt_shots += "\n"

    evaluation_data_points = []
    for data_point in load_dataset('xsum', split='test', trust_remote_code=True):
        article = data_point["document"]
        highlights = data_point["summary"]
        evaluation_data_points.append(
            EvaluationExample(
                input=prompt_shots + f"Article: {article}\nSummary:",
                output=f" {highlights}",
            )
        )
    return evaluation_data_points

def prepare_human_eval() -> List[EvaluationExample]:
    evaluation_data_points = []
    for data_point in load_dataset('openai_humaneval', split='test'):
        evaluation_data_points.append(
            EvaluationExample(
                input=data_point["prompt"],
                output=data_point["canonical_solution"],
            )
        )
    return evaluation_data_points

def prepare_gsm8k(n_shot: int = 0, seed: int = 42) -> List[EvaluationExample]:
    prompt_shots = ""
    if n_shot > 0:
        shots = load_dataset("openai/gsm8k", name="main", split="train").shuffle(seed=seed).select(range(n_shot))
        for i in range(n_shot):
            prompt = f"Question: {shots[i]['question']}\nAnswer: {shots[i]['answer']}\n"
            prompt_shots += prompt
        prompt_shots += "\n"
    
    evaluation_data_points = []
    for data_point in load_dataset("openai/gsm8k", name="main", split="test"):
        question = data_point["question"]
        answer = data_point["answer"]
        evaluation_data_points.append(
            EvaluationExample(
                input=prompt_shots + f"Question: {question}\nAnswer:",
                output=f" {answer}",
            )
        )
    return evaluation_data_points


def prepare_aqua_rat() -> List[EvaluationExample]:
    cot_prompt = """
Q: Two friends plan to walk along a 43-km trail, starting at opposite ends of the trail at the same time. If Friend P's rate is 15% faster than Friend Q's, how many kilometers will Friend P have walked when they pass each other?
Options: (a) 21 (b) 21.5 (c) 22 (d) 22.5 (e) 23
A: If Q complete x kilometers, then P completes 1.15x kilometers. x + 1.15x = 43 2.15x=43 x = 43/2.15 = 20 Then P will have have walked 1.15*20=23 km. So the answer is (e).

Q: In the coordinate plane, points (x, 1) and (5, y) are on line k. If line k passes through the origin and has slope 1/5, then what are the values of x and y respectively?
Options: (a) 4 and 1 (b) 1 and 5 (c) 5 and 1 (d) 3 and 5 (e) 5 and 3
A: Line k passes through the origin and has slope 1/5 means that its equation is y=1/5*x. Thus: (x, 1)=(5, 1) and (5, y) = (5,1) -->x=5 and y=1. So the answer is (c).

Q: There are k-2 members in a certain band, including Jim and Ellen. Two members are to be selected to attend the Grammy awards ceremony. If there are 6 possible combinations in which Jim and Ellen are not selected, what is the value of k?
Options: (a) 8 (b) 9 (c) 10 (d) 11 (e) 12
A: There are k-2 members in the band, and k-4 members without Jim and Ellen. (k-4)C2 = 6 (k-4)(k-5)/2 = 6 (k-4)(k-5) = 12 = 4*3 k = 8. So the answer is (c).

Q: The speed at which a man can row a boat in still water is 25 kmph. If he rows downstream, where the speed of current is 11 kmph, what time will he take to cover 80 metres?
Options: (a) 18 seconds (b) 27 seconds (c) 26 seconds (d) 12 seconds (e) 8 seconds
A: Speed of the boat downstream = 25 +11 = 36 kmph = 36 * 5/18 = 10 m/s Hence time taken to cover 80 m = 80/10 = 8 seconds. So the answer is (e).

Q: There are k-2 members in a certain band, including Jim and Ellen. Two members are to be selected to attend the Grammy awards ceremony. If there are 6 possible combinations in which Jim and Ellen are not selected, what is the value of k?
Options: (a) 8 (b) 9 (c) 10 (d) 11 (e) 12
A: There are k-2 members in the band, and k-4 members without Jim and Ellen. (k-4)C2 = 6 (k-4)(k-5)/2 = 6 (k-4)(k-5) = 12 = 4*3 k = 8. So the answer is (a).
    """

    dataset = load_dataset("deepmind/aqua_rat", split="test")

    evaluation_data_points = []
    for data_point in dataset:
        options = data_point["options"]
        formatted_options = "(a) {} (b) {} (c) {} (d) {} (e) {}".format(
            options[0][2:], options[1][2:], options[2][2:], options[3][2:], options[4][2:]
        )

        question_prompt = f"{cot_prompt}\nQ: {data_point['question']}\nOptions: {formatted_options}\nA:"
        answer = f" {data_point['correct']}"

        evaluation_data_points.append(
            EvaluationExample(
                input=question_prompt,
                output=answer,
            )
        )

    return evaluation_data_points


def prepare_finance_alpaca() -> List[EvaluationExample]:
    evaluation_data_points = []
    system_prompt = "You are a finance expert. Answer the following questions to the best of your knowledge, and explain as much as possible."
    for data_point in load_dataset("candenizkocak/finance-alpaca", split='train'):
        evaluation_data_points.append(
            EvaluationExample(
                input=f'{system_prompt}\n{data_point["instruction"]}',
                output=data_point["output"],
            )
        )
    return evaluation_data_points


def prepare_alpaca() -> List[EvaluationExample]:
    evaluation_data_points = []
    for data_point in load_dataset("tatsu-lab/alpaca", split='train'):
        # Check if 'input' is non-empty or empty
        if data_point["input"]:
            prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{data_point["instruction"]}
### Input:
{data_point["input"]}
### Response:
"""
        else:
            prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
{data_point["instruction"]}
### Response:
"""
        evaluation_data_points.append(
            EvaluationExample(
                input=prompt,
                output=data_point["output"],
            )
        )
    return evaluation_data_points


def prepare_wmt14_de_en_translation() -> List[EvaluationExample]:
    evaluation_data_points = []
    system_prompt = "You are a professional translator. Translate the following German sentence into English."

    dataset = load_dataset("wmt/wmt14", "de-en", split="test")

    for data_point in dataset:
        german_text = data_point["translation"]["de"]
        english_translation = data_point["translation"]["en"]
        evaluation_data_points.append(
            EvaluationExample(
                input=f"{system_prompt}\nGerman: {german_text}",
                output=english_translation,
            )
        )

    return evaluation_data_points


def prepare_custom(data_path: str, prompt_field: str = "prompt", response_field: str = "response") -> List[EvaluationExample]:
    evaluation_data_points = []
    for _, data_point in pd.read_json(data_path, lines=True).iterrows():
        evaluation_data_points.append(
            EvaluationExample(
                input=data_point[prompt_field],
                output=data_point[response_field],
            )
        )
    return evaluation_data_points

def get_data(
    random_shuffle: bool,
    num_samples: int,
    dataset: str,
    data_path: Optional[str] = None,
    n_shot: int = 0,
    seed: int = 42,
    prompt_field: str = "prompt",
    response_field: str = "response",
) -> List[EvaluationExample]:
    if dataset == DatasetFormat.CHAT_FORMAT:
        evaluation_data_points = prepare_evaluation_examples_chat_format(data_path)
    elif dataset == DatasetFormat.CNN_DM_SUMMARIZATION:
        evaluation_data_points = prepare_cnn_dm_summarization_format(n_shot=n_shot)
    elif dataset == DatasetFormat.XSUM_SUMMARIZATION:
        evaluation_data_points = prepare_xsum_summarization_format(n_shot=n_shot)
    elif dataset == DatasetFormat.CNN_DM_LM:
        evaluation_data_points = prepare_cnn_dm_lm_format()
    elif dataset == DatasetFormat.HUMAN_EVAL:
        evaluation_data_points = prepare_human_eval()
    elif dataset == DatasetFormat.GSM8k:
        evaluation_data_points = prepare_gsm8k(n_shot=n_shot)
    elif dataset == DatasetFormat.AQUA_RAT:
        evaluation_data_points = prepare_aqua_rat()
    elif dataset == DatasetFormat.FINANCE_ALPACA:
        evaluation_data_points = prepare_finance_alpaca()
    elif dataset == DatasetFormat.WMT14_DE_EN:
        evaluation_data_points = prepare_wmt14_de_en_translation()
    elif dataset == DatasetFormat.ALPACA:
        evaluation_data_points = prepare_alpaca()
    elif dataset == DatasetFormat.CUSTOM_JSONL:
        evaluation_data_points = prepare_custom(data_path, prompt_field=prompt_field, response_field=response_field)
    else:
        raise NotImplementedError(f"Unknown dataset format {dataset}")
    
    if random_shuffle:
        random.shuffle(evaluation_data_points)

    if num_samples:
        evaluation_data_points = evaluation_data_points[:num_samples]

    return evaluation_data_points

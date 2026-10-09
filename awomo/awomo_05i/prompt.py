# Copyright (c) 2026 Awomo-WAM Team. Licensed under the Apache License, Version 2.0.
"""Qwen3-VL prompt for Awomo-0.5I."""

from __future__ import annotations

import numpy as np
import torch

QUESTION = "What should the robot do now? Answer with one short imperative sentence."
IMAGE_PAD = "<|image_pad|>"


def build_messages(layout_sentence: str, task: str, num_history: int) -> list[dict]:
    task = str(task).strip()
    if not task:
        raise ValueError("task instruction must be non-empty.")
    content: list[dict] = [{"type": "text", "text": layout_sentence + "\n"}]
    if num_history > 0:
        content.append({"type": "text", "text": f"Past {num_history} frames, 2 s apart, oldest first: "})
        content.extend({"type": "image"} for _ in range(num_history))
        content.append({"type": "text", "text": "\n"})
    content.append({"type": "text", "text": "Current frame: "})
    content.append({"type": "image"})
    content.append({"type": "text", "text": f"\nTask: {task}\n{QUESTION}"})
    return [{"role": "user", "content": content}]


class PromptEncoder:
    """(task, history frames, current frame) -> Qwen3-VL inputs."""

    def __init__(self, processor, layout_sentence: str) -> None:
        self.processor = processor
        self.layout_sentence = layout_sentence
        self.tokenizer = processor.tokenizer
        self.merge_size = int(processor.image_processor.merge_size)
        self.image_token_id = int(self.tokenizer.convert_tokens_to_ids(IMAGE_PAD))
        self._cache: dict = {}

    def _tokens(self, task: str, num_history: int, grid: tuple[int, int, int]):
        key = (str(task).strip(), int(num_history), tuple(int(v) for v in grid))
        if key in self._cache:
            return self._cache[key]
        text = self.processor.apply_chat_template(build_messages(self.layout_sentence, task, num_history),
                                                  tokenize=False, add_generation_prompt=True, enable_thinking=False)
        count = int(np.prod(grid)) // (self.merge_size**2)
        pieces = text.split(IMAGE_PAD)
        expanded = pieces[0]
        for piece in pieces[1:]:
            expanded += IMAGE_PAD * count + piece
        encoding = self.tokenizer(expanded, add_special_tokens=False, return_offsets_mapping=True)
        ids = torch.tensor(encoding["input_ids"], dtype=torch.long)
        task_start = expanded.rfind(f"Task: {str(task).strip()}\n") + len("Task: ")
        layout_start = expanded.find(self.layout_sentence)
        positions: list[int] = []
        for begin_char, end_char in ((layout_start, layout_start + len(self.layout_sentence)),
                                     (task_start, task_start + len(str(task).strip()))):
            positions.extend(index for index, (begin, end) in enumerate(encoding["offset_mapping"])
                             if end > begin_char and begin < end_char)
        self._cache[key] = (ids, torch.tensor(positions, dtype=torch.long))
        return self._cache[key]

    def encode(self, task: str, current: np.ndarray, history: list[np.ndarray]) -> dict[str, torch.Tensor]:
        """current / history: [H, W, 3] float in [0, 1]; history holds valid frames only, oldest first."""
        images = [torch.as_tensor(np.asarray(f, dtype=np.float32)).permute(2, 0, 1).contiguous()
                  for f in [*history, current]]
        out = self.processor.image_processor(images=images, do_resize=False, do_rescale=False, return_tensors="pt")
        grids = out["image_grid_thw"]
        ids, condition_pos = self._tokens(task, len(history), tuple(int(v) for v in grids[-1].tolist()))
        return {
            "input_ids": ids[None],
            "attention_mask": torch.ones_like(ids)[None],
            "mm_token_type_ids": (ids == self.image_token_id).long()[None],
            "condition_pos": condition_pos[None],
            "summary_pos": torch.tensor([ids.numel() - 1], dtype=torch.long),
            "pixel_values": out["pixel_values"],
            "image_grid_thw": grids,
        }

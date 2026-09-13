"""Shared helpers for the data packing tools."""

from pathlib import Path

import binpacking


def pack_data(data_list: list, pack_length: int, batch_size: int = 1024) -> list:
    """Pack samples into bins of at most pack_length tokens, batched to bound memory."""
    all_packed = []
    for i in range(0, len(data_list), batch_size):
        batch = data_list[i : i + batch_size]
        lengths = [d["num_tokens"] for d in batch]
        grouped = binpacking.to_constant_volume(list(enumerate(lengths)), pack_length, weight_pos=1)
        for group in grouped:
            group_data = []
            for idx, _ in group:
                item = batch[idx]
                group_data.append(
                    {
                        "image": item["image"],
                        "question": item["question"],
                        "answer": item["answer"],
                        "num_tokens": item["num_tokens"],
                    }
                )
            all_packed.append(group_data)
    return all_packed


def read_input_list(txt_path: Path) -> list[Path]:
    """Read a txt file containing one JSONL file path per line.
    Empty lines and lines starting with '#' are ignored."""
    paths = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                paths.append(Path(line))
    return paths

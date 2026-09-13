# Data Format & Packing Pipeline

[中文阅读](./data_format_zh.md)

## Overview

Training expects **packed** JSONL: each line contains multiple original samples concatenated into a single sequence up to `pack_length` tokens. This maximizes GPU utilization by removing padding waste.

The pipeline: raw JSONL → tokenize + count → pack → training-ready JSONL.

---

## 1. Raw OCR JSONL Schema

Each line in a raw JSONL file is one training sample. `normalize_sample()` accepts two formats:
Format A when the sample carries a `conv` field, Format B otherwise. Both are normalized to an
`image` / `question` / `answer` triple.

| Field                     | Type                 | Format | Notes                                                                                                                                                       |
| ------------------------- | -------------------- | ------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `img_path` / `image_path` | `str` \| `list[str]` | A + B  | Absolute image path. Either key is accepted; a list takes its first entry.                                                                                  |
| `conv`                    | `list[dict]`         | A      | The 0-th turn's `question` / `answer` is used.                                                                                                              |
| `conversations`           | `list[dict]`         | B      | Alternating `human`/`gpt` (or `user`/`assistant`) turns. The `<image>` placeholder in the `human` value is stripped and the remainder becomes the question. |

### Format A

```json
{
  "img_path": "/absolute/path/to/image.png",
  "conv": [
    {
      "question": "Extract all body text from the document image as markdown...",
      "answer": "# Title\n\nBody text ..."
    }
  ]
}
```

### Format B

```json
{
  "image_path": ["/absolute/path/to/image.png"],
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nExtract all body text from the document image as markdown..."
    },
    { "from": "gpt", "value": "# Title\n\nBody text ..." }
  ]
}
```

## 2. Packing Pipeline

The `tools/pipeline_count_and_pack.py` script does two things:

1. **Count phase** (parallel): tokenizes each sample and writes token counts to `count_output_dir/*.jsonl`
2. **Pack phase**: greedily packs samples up to `pack_length` tokens per output line (First-Fit Decreasing)

### Config file: `configs/data_list.txt`

Plain text file, one path per line:

```
/path/to/dataset_A.jsonl
/path/to/dataset_B.jsonl
/path/to/dataset_C.jsonl
# Comments starting with # are ignored
```

### Run

```bash
MODEL_PATH=/path/to/HunyuanOCR \
INPUT_LIST=./configs/data_list.txt \
PACK_OUTPUT=./data/parsing_packed_20480.jsonl \
    bash scripts/pack_data.sh
```

### Environment variables

| Env var               |                                  Default | Description                                           |
| --------------------- | ---------------------------------------: | ----------------------------------------------------- |
| `MODEL_PATH`          |                             _(required)_ | HunyuanOCR base model dir (for tokenizer + processor) |
| `INPUT_LIST`          |                `./configs/data_list.txt` | Path to file listing raw JSONLs                       |
| `COUNT_OUTPUT_DIR`    |             `./data/parsing_jsonl_count` | Temp dir for count-phase output                       |
| `PACK_OUTPUT`         | `./data/parsing_packed_{PACK_LEN}.jsonl` | Final packed JSONL                                    |
| `PACK_LEN`            |                                  `20480` | Max sequence length per packed line                   |
| `NUM_PROCESSES`       |                                     `32` | Multiprocess count workers                            |
| `THREADS_PER_PROCESS` |                                      `8` | Threads per count worker                              |
| `LOG_FILE`            |                          `pack_data.log` | Progress log path                                     |

### Output schema

The output is a packed JSONL file, where each line is a JSON array of training samples:

```json
[
    {
        "image": "/abs/path/1.png",
        "question": "Extract the text from the image",
        "answer": "Recognized document text",
        "num_tokens": 4123
    },
    {
        "image": "/abs/path/2.png",
        "question": "Extract the text from the image",
        "answer": "Recognized document text",
        "num_tokens": 4444
    }
]
```

The sum of `num_tokens` in one array is at most `pack_length`. `cu_seqlens` is recomputed by the training collator and is not stored in the JSONL.

### Performance tips

- CPU-bound: scales linearly with `NUM_PROCESSES`. On 128-core machine, use `NUM_PROCESSES=32~64`.
- Image data does **not** need to be pre-loaded — the count phase only tokenizes text; images are loaded lazily during training.
- First run generates count cache; re-running with the same input list skips the count phase.

## 3. Verifying Packed Data

Sanity check the packed JSONL:

```bash
python -c "
import json
with open('./data/parsing_packed_20480.jsonl') as f:
    for i, line in enumerate(f):
        pack = json.loads(line)
        print(f'pack {i}: {len(pack)} samples, '
              f'{sum(item.get(\"num_tokens\", 0) for item in pack)} tokens')
        if i >= 3: break
"
```

Every pack should land close to `pack_length=20480`:

```
pack 0: 7 samples, 20438 tokens
pack 1: 5 samples, 19821 tokens
pack 2: 12 samples, 20301 tokens
...
```

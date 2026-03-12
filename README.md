# PruneSID: Prune Redundancy, Preserve Essence

[![arXiv](https://img.shields.io/badge/arXiv-2603.09480-b31b1b.svg)](https://arxiv.org/abs/2603.09480)

This repository contains the official implementation of **"Prune Redundancy, Preserve Essence: Vision Token Compression in VLMs via Synergistic Importance-Diversity"**, which has been **accepted to ICLR 2026**.

## 📢 News

- **2026-03**: Our paper *"Prune Redundancy, Preserve Essence: Vision Token Compression in VLMs via Synergistic Importance-Diversity"* is **accepted to ICLR 2026**.


## 🔧 Usage

Below we provide example commands to reproduce PruneSID on `LLaVA` and `Qwen2-VL`.

## 1️⃣ Test on LLaVA.

1. First install the [LLaVA](https://github.com/haotian-liu/LLaVA) environment.
  ```bash
    git clone https://github.com/haotian-liu/LLaVA.git
    cd LLaVA
    conda create -n llava python=3.10 -y
    conda activate llava
    pip install -e .
    cd ..
  ```
2. Then install [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) package.
  ```bash
    cd lmms-eval
    uv pip install -e .
    cd ..
  ```
3. Inference examples:
  **LLaVA-1.5**
    **LLaVA-NeXT**

## 2️⃣ Test on Qwen2-VL.

1. First install [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) package.
2. Then install [qwen2-vl](https://github.com/xwjim/Qwen2-VL) environment.
  ```bash
    pip install git+https://github.com/huggingface/transformers@21fac7abba2a37fae86106f87fcf9974fd1e3830
    pip install accelerate
    pip install qwen-vl-utils
  ```
3. Inference example:
  ```bash
    CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 --main_process_port=12345 -m lmms_eval \
      --model qwen2_vl \
      --model_args=pretrained=Qwen/Qwen2-VL-7B-Instruct,max_pixels=2359296 \
      --tasks mme \
      --batch_size 1 \
      --log_samples \
      --log_samples_suffix reproduce \
      --output_path ./logs/ \
      --plug_in_model prunesid_qwen2 \
      --need_token_num 64
  ```

## 📚 Citation

If you use PruneSID or find our paper useful in your research, please cite:

```bibtex
@inproceedings{
fang2026prune,
  title={Prune Redundancy, Preserve Essence: Vision Token Compression in {VLM}s via Synergistic Importance-Diversity},
  author={Zhengyao Fang and Pengyuan Lyu and Chengquan Zhang and Guangming Lu and Jun Yu and Wenjie Pei},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=i36E5Ezm0H}
}
```

We will update the BibTeX entry with publication details once they are available.

## 📜 License

This project is licensed under the **Apache License 2.0**.  
You may not use this file except in compliance with the License.  
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

For the full license text, please refer to the official Apache 2.0 license:  
`http://www.apache.org/licenses/LICENSE-2.0`
<div align="center">
<img src="rl/open-instruct/assets/dr_tulu_logo.png" alt="DR Tulu" width="500"/>

# DR Tulu: Reinforcement Learning with Evolving Rubrics for Deep Research


[**Paper**](https://allenai.org/papers/drtulu) • [**Data & Models**](https://huggingface.co/collections/rl-research/dr-tulu) • [**Blogpost**](http://allenai.org/blog/dr-tulu) • [**Video**](https://youtu.be/4i0W9qAf8K8)• [**Interactive Demo**](https://www.dr-tulu.org)
</div>

DR Tulu-8B is the first open Deep Research (DR) model trained for long-form DR tasks. DR Tulu-8B matches OpenAI DR on long-form DR benchmarks.

<div align="center">
<img src="assets/rler_teaser.png" alt="DR Tulu Overview" width="800"/>
</div>

---

## Release Notes 
- Feburary 9, 2026: 🔥 We released a free interactive demo for DR Tulu-8B! Try it out at [dr-tulu.org](https://www.dr-tulu.org/chat)!
- November 19, 2025: Initial code release.
- November 25, 2025: We released our interactive CLI demo code, along with additional documentation for evaluation, training, and our new RL checkpoints.

## Overview

This repository contains three main components:

- **[`agent/`](agent/)**: SWE-agent rollout code plus the shared `agent_rl`
  training/runtime interfaces.

- **[`rl/`](rl/open-instruct/)**: RL training code based on [Open-Instruct](https://github.com/allenai/open-instruct) for training deep research agents with GRPO and evolving rubrics.

- **[`sft/`](sft/llama-factory/)**: SFT training code based on [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for supervised fine-tuning of deep research agents.

For detailed setup and usage instructions, see the README files in each subdirectory.

---

## Agent setup

Install the current SWE-agent package from `agent/`:

```bash
cd agent
uv pip install -e .
```

See [`agent/README.md`](agent/README.md) for the maintained rollout entry points.

---

## Training

### Supervised Fine-Tuning (SFT)

For supervised fine-tuning of deep research agents using high-quality demonstration data:

```bash
cd sft/llama-factory/
# See sft/llama-factory/README.md for detailed instructions
```

See [`sft/llama-factory/README.md`](sft/llama-factory/) for complete SFT training setup and configuration.

### Reinforcement Learning (RL)

For training deep research agents with GRPO and evolving rubrics:

```bash
cd rl/open-instruct/
# See rl/open-instruct/README.md for detailed instructions
```

See [`rl/open-instruct/README.md`](rl/open-instruct/) for complete RL training setup, including reward model training and policy optimization.

---

## Acknowledgments

DR Tulu is provided by The Allen Institute for Artificial Intelligence (Ai2). The code for this project is developed in collaboration with student researchers at the University of Washington, Carnegie Mellon University, and MIT.

---

## Citation and Contact

If you find our work useful, please cite:

```bibtex
@article{shao2025dr,
  title={DR Tulu: Reinforcement Learning with Evolving Rubrics for Deep Research},
  author={Shao, Rulin and Asai, Akari and Shen, Shannon Zejiang and Ivison, Hamish and Kishore, Varsha and Zhuo, Jingming and Zhao, Xinran and Park, Molly and Finlayson, Samuel G and Sontag, David and others},
  journal={arXiv preprint arXiv:2511.19399},
  year={2025}
}
```
If you have any questions, you can contact [Rulin Shao](https://rulinshao.github.io/), [Akari Asai](https://akariasai.github.io/), [Shannon Shen](https://www.szj.io/), and [Hamish Ivison](https://ivison.id.au/) or open a github issue. 

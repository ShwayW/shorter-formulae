# Synthesizing compact physics formulae

## 1. Installation

Python ≥ 3.10 and a C++20 compiler.

```bash
pip install torch numpy scipy pandas matplotlib scikit-learn sympy tqdm joblib h5py \
            numexpr sympytorch peft pytorch_lightning omegaconf hydra-core ordered_set \
            dataclass_dict_convert pyarrow huggingface_hub
```

Compile the C++ evaluator for Polish-notation formulae (run from this directory):

```bash
g++ -std=c++20 -O2 -fPIC -shared -o extGetHeadAndInputs.so cpp/extGetHeadAndInputs.cpp
g++ -std=c++20 -O2 -fPIC -shared -o extGetSubfmlByInd.so   cpp/extGetSubfmlByInd.cpp
g++ -std=c++20 -O2 -fPIC -shared -o extEvalPN.so           cpp/extEvalPN.cpp
```

## 2. Datasets

**SRBench (Feynman):** the `feynman_*` datasets of the Penn Machine Learning Benchmarks,
<https://github.com/EpistasisLab/pmlb>. Place them in `datasets/pmlb/`:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/EpistasisLab/pmlb datasets/pmlb
(cd datasets/pmlb && git lfs pull --include="datasets/feynman_*")
```

**LLM-SRBench:** <https://huggingface.co/datasets/nnheui/llm-srbench>. Place it in
`datasets/llmsrbench/`:

```bash
python download_llmsrbench.py
```

## 3. Third-Party Model Weights

| Method | Weights | Place at |
|---|---|---|
| E2E (Kamienny et al., 2022), also used by TPSR | <https://dl.fbaipublicfiles.com/symbolicregression/model1.pt> | `weights/e2e/e2e_model.pt` and `TPSR/symbolicregression/weights/model.pt` |
| NeSymReS (Biggio et al., 2021), 100M model | <https://github.com/SymposiumOrganization/NeuralSymbolicRegressionThatScales> (pretrained weights linked in its README) | `weights/nesymres/100M.ckpt` |
| PhyE2E (Ying et al., 2025) | <https://drive.google.com/drive/folders/14M0Ed0gvSKmtuTOornfEoup8l48IfEUW> | `weights/phye2e/phye2e_model.pt` |
| SymFormer (Vastl et al., 2022) | <https://data.ciirc.cvut.cz/public/projects/2022SymFormer/checkpoints/symformer-univariate.tar.gz>, <https://data.ciirc.cvut.cz/public/projects/2022SymFormer/checkpoints/symformer-bivariate.tar.gz> | `weights/symformer/` |
| Transformer4SR (Lalande et al., 2023) | `best_model_weights/` of <https://github.com/omron-sinicx/transformer4sr> | `tf4sr/best_model_weights/` |
| Llama-3.1-8B-Instruct (LLM baselines) | <https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct> | served with vLLM, see `configs/` |
| Llama-3.3-70B-Instruct (LLM baselines) | <https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct> | served with vLLM, see `configs/` |
| Qwen2.5-Coder-32B-Instruct (LLM baselines) | <https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct> | served with vLLM, see `configs/` |

## 4. Our Model Weights

The weights of our six transformers are hosted on Hugging Face at
<https://huggingface.co/ShwayW/shorter-formulae>. Download them into `checkpoints/` here,
so that each model sits in its own `checkpoints/<name>_res/` folder:

```bash
hf download ShwayW/shorter-formulae --local-dir checkpoints
```

`checkpoints/readme` (and the model card on Hugging Face) describes each model and how to
evaluate it.

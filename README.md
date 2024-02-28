Repository for our experiments on inflicting "Brain Damage" to an LLM [as described in our blog](https://csg.ziti.uni-heidelberg.de/blog/llm-brain-damage/).
Based on the [https://huggingface.co/google/gemma-7b-it](gemma-7b-it model) published by Google.
To run the experiments, drop the [huggingface transformers version of the model](https://huggingface.co/google/gemma-7b-it/tree/main) in the folder "models".<br>
To run the experiment, first set up the conda environment and activate it by running

```
conda create --name llm_brain_damage --file requirements.txt
conda activate llm_brain_damage
```

in the repository root folder.
This implementation is using flash attention by default, which has to be installed seperately:

```
pip install flash-attn
```

Please note that the fliptensors code currently contains a memory leak, which may cause the model to crash after a small number of inputs, depending on the size of your GPU memory. We are currently working on fixing this issue.

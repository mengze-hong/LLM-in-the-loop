<div align="center">
  <a href="https://swarms.world">
    <h1>Empirical Study: LLM-native Text Clustering</h1>
  </a>
</div>
<p align="center">
  <em>The goal of this experiment is to fill in the gap of missing LLM-native results (i.e., direct prompting) in text clustering.</em>
</p>

----

## Install
```bash
pip install -r requirements.txt
```

### To run LLM-native clustering with varying prompts

```bash
clustering_prompt_clinc150.ipynb
clustering_prompt_banking77.ipynb
clustering_prompt_hwu64.ipynb
```

Fill in the OpenAI key in:

```python
client = OpenAI(
    api_key="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
)
```

### To run LLM-native clustering with varying task configurations
```bash
clustering_n_k.ipynb
```

The results are stored in `./clustering_result/`, and summarized result statistics can be found in `./clustering_result/count_statistics`.

### To run PromptWizard for prompt tuning
Replace the `prompt_pool.yaml` in the PromptWizard library with `./PromptWizard-clustering/prompt_pool.yaml`.
Fill in the OpenAI key in `./PromptWizard-clustering/.env`.

```bash
cd PromptWizard-clustering
demo.ipynb
```

To configure the generated prompt (e.g., with or without reasoning steps), modify the parameters in:

```python
best_prompt, expert_profile = gp.get_best_prompt(use_examples=True, run_without_train_examples=False, generate_synthetic_examples=False)
```
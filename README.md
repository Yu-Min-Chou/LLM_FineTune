# LLM_FineTune
Share the methods about Parameter-Efficient Fine-Tuning and the tricks for running LLM on consumer-level GPUs

------------------

## Fine Tuning Methos
Currently, transformers-based Large Language Model(LLM) have achieve state-of-the-art performance in every tasks of NLP. Generally, LLM models are pre-trained with specially designed pre-training tasks on large-scale unlabeled datasets. These pre-trained models are very effective as general-purpose semantic features,  which which have largely raised the performance bar of NLP tasks. Subsequently, fine-tuning a pre-trained language model on downstream tasks has become a paradigm for NLP tasks. 

### LoRA

### Prompt Tuning

### P-Tuning

### Prefix-Tuning

### (IA)3

| Fine-Tuning Method | Batch Size | Number of total params | Trainable params | Required GPU memory | Speed (Train) | Speed(Eval) | Accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LoRA | 8 | 1,231,940,608 | 2,359,296 (0.191%) | 9392MiB | 3.32s/it | 8.96s/it |96.47% |
| Prompt Tuning | 8 | 1,229,622,272 | 40,960 (0.003%) | 9768MiB | 3.81s/it | 9.51s/it |87.24% |
| P-Tuning | 8 | 1,229,902,080 | 320,768 (0.026%) |9800MiB | 3.68s/it | 9.48s/it |96.47% |
| Prefix-Tuning | 8 | 1,230,564,352 | 983,040 (0.079%) |9392MiB | 3.32s/it | 8.96s/it |96.47% |
| (IA)3 | 8 | 1,229,863,936 | 282,624 (0.022%) | 9392MiB | 3.32s/it | 8.96s/it |96.47% |
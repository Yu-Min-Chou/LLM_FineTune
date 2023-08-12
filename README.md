# LLM_FineTune
Share the methods about Parameter-Efficient Fine-Tuning and the tricks for running LLM on consumer-level GPUs

------------------

## Fine Tuning Methods
Currently, transformers-based Large Language Model(LLM) have achieve state-of-the-art performance in every tasks of NLP. Generally, LLM models are pre-trained with specially designed pre-training tasks on large-scale unlabeled datasets. These pre-trained models are very effective as general-purpose semantic features,  which which have largely raised the performance bar of NLP tasks. Subsequently, fine-tuning a pre-trained language model on downstream tasks has become a paradigm for NLP tasks. However, As models get larger(3B~176B), full fine-tuning of the model on consumer-grade hardware becomes infeasible. Motivated by that, research community proposed Parameter-Efficient Fine-Tuning (PEFT), which enable efficient adaptation of pre-trained language models (PLMs) to various downstream applications without fine-tuning all the model's parameters.

This project aims to briefly explain the various PEFT technologies and provide sample codes. Based on the paper: [Scaling Down to Scale Up: A Guide to Parameter-Efficient Fine-Tuning](https://arxiv.org/pdf/2303.15647.pdf), PEFT methods can be divided into three categories as follows
  - **Additve method**: The main idea behind additive methods is augmenting the existing pre-trained model with extra
parameters or layers and training only the newly added parameters. Moreover, additive methods can be divided into Adapters and Soft Prompts according to adding new layers or modifying input text in the model.
  - **Selective method**: The main idea behind selective method is selecting a subset of pre-trained model to update. For example, fine-tuning only a few top layers of a network.
  - **Reparametrization-based method**: Reparametrization-based methods leverage low-rank representation to minimize the number of trainable parameters. LoRA is the one of the most common method of this category.
![Scaling Down to Scale Up: A Guide to Parameter-Efficient Fine-Tuning](https://github.com/Yu-Min-Chou/LLM_FineTune/assets/42434345/74b5662f-878d-4644-9481-7f961eab0d3c)

Here, we only introduce a few of the methods which have been supported by huggingface🤗 [peft](https://huggingface.co/docs/peft/main/en/index)

### LoRA: [Low-Rank Adapatation of Large Launguage Models](https://arxiv.org/abs/2106.09685)
Neural networks contain many fully connected layers, which are implemented with the help of matrix multiplication. Recent research consider that the parameter matrix for weight update can be effectively learned despite being randomly projected into a smaller subspace. Based on the insight, LoRA reduces the number of parameters and the operations of matrix multiplication by **low-rank decomposition**.

Specifically, for a pre-trained wegiht matrix $W_{0} \in R^{d\times k}$, it's update can be represented as a low-rank composition $W_{0} + \Delta W = W_{0} + BA$, where $B \in R^{d\times r}, A \in R^{r\times k}, r << max(d,k)$, such that the size of matrix decreases from $d \times k$ to $d \times r + r \times k$. During training process, pre-trained model weights $W_{0}$ are freezed and we only optimize $A$ and $B$. 

We encourage readers to read the paper for further detail about the following questions
  - How to choose the rank $r$
  - Which and how many weight matrix should be decomposed
  - Can LoRA outperfrom full fine-tuning in terms of model quality

![image](https://github.com/Yu-Min-Chou/LLM_FineTune/assets/42434345/a1e2093a-d28d-4dd0-8543-cd87e6f7a07a)

### Prompt Tuning: [The Power of Scale for Parameter-Efficient Prompt Tuning](https://arxiv.org/abs/2104.08691)
For LLM, prompt can greatly affect the performance on various task. Motivated by that, prompt tuning aims to find the prompts which release the potential of LLMs and achieve well performance on downstream tasks. The naive methods of prompt tuning is using manually designed prompts. However, the naive methods lack flexibility and attain inferior results. To solve that, the authors propose to learn prompts by updating parameters through backpropagation, instead of manually designing prompts.

Prompt tuning first initializes a task-specific prompt, which is represented as embedding matrix $P_{e} \in R^{p\times e}$, where $p$ is the length of the prompt. Then we prepend them to the inputs $X_{e} \in R^{n\times e}$, which is embed from a series of $n$ tokens $x_1, x_2, ..., x_n$. During training, the concatenated input $[P_e;X_e]$ flows though the LLM as normal, and only the prompt $P_{e}$ is updated.

![image](https://github.com/Yu-Min-Chou/LLM_FineTune/assets/42434345/75dcec4c-b6a2-48f6-975f-5aefa4b3027d)

We encourage readers to read the paper for further detail about the following questions
 - What should be the length of the prompt token $p$?

### P-Tuning: [GPT Understands, Too](https://arxiv.org/abs/2103.10385)
P-Tuning share the similiar insight with prompt tuning: utilizing a suitable prompt to enhance the LLM performance. There are two main difference between the prompt tuning and p-tuning. First, p-tuning learns a prompt encoder to encode the pseudo prompts, which is initialized with unused words or task name, instead of directly learning prompts itself. Second, encoded pseudo tokens can be inserted at any position instead of only prefix.

To be a step further, learning encoder can solves two two optimization challenges of learning prompts.
  - Word embedding layers of pretrained LLMs have become highly discrete, randomly initializing the prompts easily lead to local optimal
  - Intuitively, the authors believe the values of prompt embeddings $h_i$ should be dependent on each other rather than independent
    
Motivated by above two concerns, the encoder comprises a bidirectional LSTM, along with a ReLU-activated two-layer multilayer perceptron (MLP). In this way, the structure of LSTM improves the integrity of prompt, and the MLP with Relu activation function encourage discreteness.

![image](https://github.com/Yu-Min-Chou/LLM_FineTune/assets/42434345/d9042715-1625-461e-a124-442881073eed)

### Prefix-Tuning: [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://aclanthology.org/2021.acl-long.353/)

### $(IA)^3$ : [Infused Adapter by Inhibiting and Amplifying Inner Activations](https://arxiv.org/abs/2205.05638)

| Fine-Tuning Method | Batch Size | Number of total params | Trainable params | Required GPU memory | Speed (Train) | Speed(Eval) | Accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LoRA | 8 | 1,231,940,608 | 2,359,296 (0.191%) | 9392MiB | 3.32it/s | 8.96it/s |96.47% |
| Prompt Tuning | 8 | 1,229,622,272 | 40,960 (0.003%) | 9768MiB | 3.81it/s | 9.51it/s |87.24% |
| P-Tuning | 8 | 1,229,902,080 | 320,768 (0.026%) |9800MiB | 3.79it/s | 9.48it/s |60.59% |
| Prefix-Tuning | 8 | 1,230,564,352 | 983,040 (0.079%) |6144MiB | 5.67it/s | 10.07it/s |96.03% |
| (IA)3 | 8 | 1,229,863,936 | 282,624 (0.022%) | 9658MiB | 3.83it/s | 9.88it/s |96.47% |

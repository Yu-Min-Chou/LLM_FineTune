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
Compared with prompt tuning and p-tuning, prefix tuning appends trainable tensors to each transformer block instead of only the input embeddings. Furthermore, the author empirically found that directly updating the parameter of learnable prompt leads to unstable optimization and a drop in performance. To solve that, prefix-tuning obtain the prompt embedding vis fully connected layers(a multilayer perceptron with two layers and a activation function in between), as illustrated in the below figure. Once training is complete, the fully connected layers are removed, and only the prompt embedding needs to be saved.

![image](https://github.com/Yu-Min-Chou/LLM_FineTune/assets/42434345/fcae8f97-1107-44cd-944c-8ed1247a0bac)

Let's dig deeper into prefix tuning vs prompt tuning. 
  - From the aspect of number of trainable parameters, prefix tuning modifies more layers of the LLM by inserting a task-specific prefix to the input sequence, while prompt tuning only append prompt embedding to the input layer. Thus, prefix-tuning requires more parameters to be finetuned. This may make prompt tuning more parameter-efficient than prefix tuning, but it may has more limited capacity to adapt to the target task than prefix training.
  - From the aspect of performance, it is reasonable to expect that prefix tuning might perform better since it has more parameters to adjust the model to the new task. However, oberviously, it brings the costs of computational resources.

### $(IA)^3$ : [Infused Adapter by Inhibiting and Amplifying Inner Activations](https://arxiv.org/abs/2205.05638)
The above methods, including prompt tuning, p-tuning, and prefix-tuning, share a shortcut: they can not perform inference for multi-task in the same batch. Compared with them, $(IA)^3$ weights the model parameters by multiplying the specific tensor with a learnable vector instead of modifying prompts.

To be more specific, $(IA)^3$ suppress or amplify some activation layers of the model, including key vector, value vector, and activation layer before feedforward neural network in each transformer block, as shown in below figure.

![image](https://github.com/Yu-Min-Chou/LLM_FineTune/assets/42434345/75f43564-8db7-43d1-bf10-ed235b8beee9)

## Getting Started
All the sample codes are implemented with 🤗[huggleface]([https://github.com/huggingface/peft/tree/main)(https://huggingface.co/)]. You can modify the sample code by replace datasets or pre-trained models. The list of datasets and pre-trained models can be found here: [dataset_list](https://huggingface.co/datasets), [model_list](https://huggingface.co/models?sort=trending). 

### Requirements
Install required packages for different fine-tuning methods. Simply navigate to the folder with `requirements.txt` and run below commands. Besides `pip`, you can install all packages with `conda install` too

```bash
git clone https://github.com/Yu-Min-Chou/LLM_FineTune.git
conda create -n llm_peft python=3.11
conda activate llm_peft
cd LLM_FineTune
python -m pip install -r requirements.txt
```

### Walk through the codes
All the sample codes in the folder `peft_mt0-large_financial_phrasebank` are very similiar. Let's take `peft_lora.py` as an example and only explain the necessary part.

At first, we specific the device(GPU) and use [mt0-large](https://huggingface.co/bigscience/mt0-large)(1.23B) for training. Different fine-tuning methods should initialize different `peft_config`. Readers can refer to the [link](https://huggingface.co/docs/peft/main/en/package_reference/tuners) to learn how to set the corresponding config for different methods. Next, we load pre-trained mt0-large and wrap it with `get_peft_model()`
```bash
device = "cuda"
model_name_or_path = "bigscience/mt0-large"
tokenizer_name_or_path = "bigscience/mt0-large"

# creating model
peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
# trainable params: 2,359,296 || all params: 1,231,940,608 || trainable%: 0.19151053100118282
```

We use [financial_phrasebank-sentences_allagree](https://huggingface.co/datasets/financial_phrasebank) for training. This dataset has 2.26k rows of data. You can easily perform training with another dataset by replace the name of the datasets and its subset. However, different dataset may have different data preprocessing methods.
```bash
# loading dataset
dataset = load_dataset("financial_phrasebank", "sentences_allagree")
dataset = dataset["train"].train_test_split(test_size=0.1)
dataset["validation"] = dataset["test"]
del dataset["test"]

classes = dataset["train"].features["label"].names
dataset = dataset.map(
  lambda x: {"text_label": [classes[label] for label in x["label"]]},
  batched=True,
  num_proc=1,
)
```

The remaining parts are written in the same way as pytorch, you can easily fine-tune LLM with huggleface🤗

### Evaluation Results
The experiments were conducted on a server withs V100 GPUs, Intel Xeon E5-2620 v4 2.1GHz CPUs, and 141GB memory. We found that prefix-tuning requires the lowest memory of GPU and has the fastest training and inference speed. Besides speed, we also found that prompt-tuning and p-tuning requires higher learning rate for better accuracy. But this involves hyper-parameter tuning, which is beyond the scope of this project.
| Fine-Tuning Method | Batch Size | Number of total params | Trainable params | Required GPU memory | Speed (Train) | Speed(Eval) | Accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LoRA | 8 | 1,231,940,608 | 2,359,296 (0.191%) | 9392MiB | 3.32it/s | 8.96it/s |96.47% |
| Prompt Tuning | 8 | 1,229,622,272 | 40,960 (0.003%) | 9768MiB | 3.81it/s | 9.51it/s |87.24% |
| P-Tuning | 8 | 1,229,902,080 | 320,768 (0.026%) |9800MiB | 3.79it/s | 9.48it/s |60.59% |
| Prefix-Tuning | 8 | 1,230,564,352 | 983,040 (0.079%) |6144MiB | 5.67it/s | 10.07it/s |96.03% |
| (IA)3 | 8 | 1,229,863,936 | 282,624 (0.022%) | 9658MiB | 3.83it/s | 9.88it/s |96.47% |

### Other implementations
In addition to fine-tuning methods, I also studied a few important technologies related to fine-tuning.
## Distributed LLM training
## Quantization for training larger LLM

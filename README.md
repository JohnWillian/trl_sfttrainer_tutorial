本篇文章将介绍如何使用 `trl` 库对 **Qwen-2.5** 模型进行 **SFT LoRA** 微调。
# 1. 环境配置
建议使用虚拟环境进行配置（如 `conda`），以避免依赖冲突。如果不需要虚拟环境，可直接跳至步骤 2。
新建一个`conda`虚拟环境：
```bash
conda create -n trl-train-env python=3.10
```

激活虚拟环境：
```bash
conda activate trl-train-env
```
# 2. 依赖库安装
安装`trl`库及相关依赖：
```bash
pip install trl peft
```

版本参考：
```python
peft==0.16.0
transformers==4.52.4
trl==0.19.0
```
---
# 3. 数据准备
以 **SFTTrainer** 为例，需将数据集处理为如下格式：
```python
[
	{
		'messages': [
			{'role': 'system', 'content': 'You are a helpful assistant.'},
			{'role': 'user', 'content': 'What is the capital of France?'},
			{'role': 'assistant', 'content': 'Paris is the capital of France.'}
		]
	},
	...
]
```
- 每条数据包含 `messages` 字段，内部为多轮对话（system/user/assistant 角色）。
- 训练集（train）和验证集（eval）格式一致。


示例数据可在 [GitHub 仓库](https://github.com/JohnWillian/trl_sfttrainer_tutorial.git) 获取（欢迎 **Star 🌟**），示例数据格式如下：
```python
[
	{
		'prompt': [{'role': 'user', 'content': '...'}],
		'completion': [{'role': 'assistant', 'content': '...'}]
	},
	...
]
```

需要将其处理成**SFTTrainer**要求的格式，数据处理脚本如下：
```python
from datasets import load_dataset

# 请根据自己的数据格式自定义修改
def formatting_prompts_func(example):
    example['messages'] = example['prompt'] + example['completion']
    return example

train_dataset_name = "./data/train"  # Note: 这里是一个目录，不是文件。目录下存放训练数据，例如`train.json`
val_dataset_name = "./data/eval"     # Note: 这里是一个目录，不是文件。目录下存放验证数据，例如`eval.json`

# 读取train数据
train_dataset = load_dataset(train_dataset_name, split="train")
train_dataset = train_dataset.shuffle(seed=42)	# Shuffle 数据

# 读取eval数据
val_dataset = load_dataset(val_dataset_name, split="test")
val_dataset = val_dataset.shuffle(seed=42)	# Shuffle 数据

# Format train data
train_dataset = train_dataset.map(
    formatting_prompts_func,
    num_proc=4,
    remove_columns=['prompt', 'completion']
)

# Format eval data
val_dataset = val_dataset.map(
    formatting_prompts_func,
    num_proc=4,
    remove_columns=['prompt', 'completion']
```
> **SFTTrainer** 兼容 [**标准**](https://huggingface.co/docs/trl/en/dataset_formats#standard)及 [**对话式**](https://huggingface.co/docs/trl/en/dataset_formats#conversational) 数据集格式。当输入对话数据集时，训练器会自动对数据集应用聊天模板。
---
# 4. Chat Template调整

在**SFTConfig**中，若需启用`assistant_only_loss`，则 `chat_template` 必须包含 `{% generation %}` 标签（参考：[Train on assistant messages only](https://huggingface.co/docs/trl/en/sft_trainer#train-on-assistant-messages-only)）。原版 **Qwen-2.5** 模型的模板未包含此标签，因此需要手动修改：
```python
from transformers import AutoTokenizer

base_model = "./models/Qwen/Qwen2.5-0.5B-Instruct"    # 基座模型路径，自行下载所需模型
modified_template = """
{%- if tools %}
    {{- '<|im_start|>system\\n' }}
    {%- if messages[0]['role'] == 'system' %}
        {{- messages[0]['content'] }}
    {%- else %}
        {{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}
    {%- endif %}
    {{- "\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>" }}
    {%- for tool in tools %}
        {{- "\\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n" }}
{%- else %}
    {%- if messages[0]['role'] == 'system' %}
        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}
    {%- else %}
        {{- '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n' }}
    {%- endif %}
{%- endif %}
{%- for message in messages %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}
    {%- elif (message.role == "assistant" and not message.tool_calls) %}
    {% generation %}    {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}    {% endgeneration %}
    {%- elif message.role == "assistant" %}
        {% generation %}{{- '<|im_start|>' + message.role }}
        {%- if message.content %}
            {{- '\\n' + message.content }}
        {%- endif %}
        {%- for tool_call in message.tool_calls %}
            {%- if tool_call.function is defined %}
                {%- set tool_call = tool_call.function %}
            {%- endif %}
            {{- '\\n<tool_call>\\n{\\\"name\\\": \\\"' }}
            {{- tool_call.name }}
            {{- '\\\", \\\"arguments\\\": ' }}
            {{- tool_call.arguments | tojson }}
            {{- '}\\n</tool_call>' }}
        {%- endfor %}
        {{- '<|im_end|>\\n' }}{% endgeneration %}
    {%- elif message.role == "tool" %}
        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\\n<tool_response>\\n' }}
        {{- message.content }}
        {{- '\\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {% generation %}{{- '<|im_start|>assistant\\n' }}{% endgeneration %}
{%- endif %}
"""

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model)	# 加载Tokenizer
tokenizer.chat_template = modified_template		# 修改 Chat Template
```
> 参考：https://github.com/huggingface/transformers/issues/34172

---
# 5. LoRA 参数配置
以 `rank=8`，`alpha=16` 为例：
```python
from peft import LoraConfig

# LoRA config
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,	# 应用 dropout，防止过拟合
    bias="none",	# 默认值，不使用任何偏置
    target_modules="all-linear",	# 使用 "all-linear" 会对模型中所有线性层都添加 LoRA 适配器
    task_type="CAUSAL_LM"
)
```


- **rank**：低秩矩阵的维度，决定 LoRA 适配器的容量。值越大，模型表达能力越强，但参数和显存占用也更高，可能过拟合；值越小则更轻量，但可能欠拟合。
- **alpha**：缩放因子，控制 LoRA 更新对原始权重的影响强度。实际更新量为 $(\alpha / r) \times \Delta W$，因此 $\alpha / r$ 才是实际的学习率比例。

---
# 6. 训练参数配置

```python
from trl import SFTConfig

# Training Config
training_arguments = SFTConfig(
    assistant_only_loss=True,   # 只计算assistant的loss
    output_dir='./output/finetune/qwen25_14b_sft_lora_r8_a16',   # 自定义模型保存路径
    per_device_train_batch_size=1,  # 需要与deepspeed配置文件保持一致
    per_device_eval_batch_size=1,   # 需要与Training Config保持一致
    gradient_accumulation_steps=8,  # 需要与deepspeed配置文件保持一致
    optim="paged_adamw_8bit",   # 优化器，使用paged_adamw训练速度更快
    num_train_epochs=2,   # 训练轮数
    gradient_checkpointing=True,    # 梯度检查点，降低显存消耗
    do_eval=True,   # 是否进行评估
    eval_strategy="steps",
    eval_steps=50,  # 每50步评估一次
    save_strategy="steps",  # 保存策略为steps
    save_steps=50,  # 每50步保存一次
    logging_steps=1,    # logging
    logging_strategy="steps",
    warmup_ratio=0.03,
    learning_rate=2e-5,
    bf16=True,  # 需要与deepspeed配置文件保持一致
    lr_scheduler_type="cosine", # Default is `linear`
    max_length=4096,    # 样本序列最大长度
    label_names=["labels"],
    report_to = "tensorboard",  # show tensorboard
)
```

---

# 7. 开始训练

```python
from trl import SFTTrainer

trainer = SFTTrainer(
    model=base_model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    args=training_arguments,
    processing_class=tokenizer, # tokenizer
    peft_config=peft_config,    # lora config
)

trainer.train()
```

---

# 8. DeepSpeed 配置
若显存有限，推荐使用 DeepSpeed 进行并行训练。以 `zero_1` 配置为例：
```yaml
compute_environment: LOCAL_MACHINE
debug: false
deepspeed_config:
  deepspeed_multinode_launcher: standard
  per_device_train_batch_size: 1  # 需要与Training Config保持一致
  gradient_accumulation_steps: 8  # 需要与Training Config保持一致
  zero3_init_flag: false
  zero_stage: 1
distributed_type: DEEPSPEED
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
mixed_precision: 'bf16' # 需要与Training Config保持一致
num_machines: 1 # 1 machine
num_processes: 8  # 使用 8 GPUs，根据实际情况自行修改
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```
将配置文件保存为 `ds_zero1_config.yaml`，然后，将前面讲到的训练配置代码整理为一个文件：
```python
from transformers import AutoTokenizer
from peft import LoraConfig
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

base_model = "./models/Qwen/Qwen2.5-0.5B-Instruct"    # 基座模型路径，自行下载所需模型
train_dataset_name = "./data/train" # Note: 这里是一个目录，不是文件。目录下存放训练数据，例如`train.json`
val_dataset_name = "./data/eval"    # Note: 这里是一个目录，不是文件。目录下存放验证数据，例如`eval.json`
new_model = "../output/finetune/qwen25_05b_sft_lora_r8_a16"   # 训练好的模型保存路径

modified_template = """
{%- if tools %}
    {{- '<|im_start|>system\\n' }}
    {%- if messages[0]['role'] == 'system' %}
        {{- messages[0]['content'] }}
    {%- else %}
        {{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}
    {%- endif %}
    {{- "\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>" }}
    {%- for tool in tools %}
        {{- "\\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n" }}
{%- else %}
    {%- if messages[0]['role'] == 'system' %}
        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}
    {%- else %}
        {{- '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n' }}
    {%- endif %}
{%- endif %}
{%- for message in messages %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}
    {%- elif (message.role == "assistant" and not message.tool_calls) %}
    {% generation %}    {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}    {% endgeneration %}
    {%- elif message.role == "assistant" %}
        {% generation %}{{- '<|im_start|>' + message.role }}
        {%- if message.content %}
            {{- '\\n' + message.content }}
        {%- endif %}
        {%- for tool_call in message.tool_calls %}
            {%- if tool_call.function is defined %}
                {%- set tool_call = tool_call.function %}
            {%- endif %}
            {{- '\\n<tool_call>\\n{\\\"name\\\": \\\"' }}
            {{- tool_call.name }}
            {{- '\\\", \\\"arguments\\\": ' }}
            {{- tool_call.arguments | tojson }}
            {{- '}\\n</tool_call>' }}
        {%- endfor %}
        {{- '<|im_end|>\\n' }}{% endgeneration %}
    {%- elif message.role == "tool" %}
        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\\n<tool_response>\\n' }}
        {{- message.content }}
        {{- '\\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {% generation %}{{- '<|im_start|>assistant\\n' }}{% endgeneration %}
{%- endif %}
"""

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.chat_template = modified_template
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

# LoRA config
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM"
)

# 组合成 conversational format
# conversational format 样例：
# 'messages': [
#     {'role': 'system', 'content': 'You are a helpful assistant.'},
#     {'role': 'user', 'content': 'What is the capital of France?'},
#     {'role': 'assistant', 'content': 'Paris is the capital of France.'}
# ]
# 请根据自己的数据格式自定义修改
def formatting_prompts_func(example):
    example['messages'] = example['prompt'] + example['completion']
    return example

def main():
    # 读取train数据
    train_dataset = load_dataset(train_dataset_name, split="train")
    train_dataset = train_dataset.shuffle(seed=42)	# Shuffle 数据

    # 读取eval数据
    val_dataset = load_dataset(val_dataset_name, split="test")
    val_dataset = val_dataset.shuffle(seed=42)	# Shuffle 数据

    # Format train data
    train_dataset = train_dataset.map(
        formatting_prompts_func,
        num_proc=4,
        remove_columns=['prompt', 'completion']
    )

    # Format eval data
    val_dataset = val_dataset.map(
        formatting_prompts_func,
        num_proc=4,
        remove_columns=['prompt', 'completion']
    )

    # Training Config
    training_arguments = SFTConfig(
        assistant_only_loss=True,   # 只计算assistant的loss
        output_dir=new_model,   # 模型保存路径
        per_device_train_batch_size=1,  # 需要与deepspeed配置文件保持一致
        per_device_eval_batch_size=1,   # 需要与Training Config保持一致
        gradient_accumulation_steps=8,  # 需要与deepspeed配置文件保持一致
        optim="paged_adamw_8bit",   # 优化器，使用paged_adamw训练速度更快
        num_train_epochs=3,   # 训练轮数
        gradient_checkpointing=True,    # 梯度检查点，降低显存消耗
        do_eval=True,   # 是否进行评估
        eval_strategy="steps",
        eval_steps=50,  # 每50步评估一次
        save_strategy="steps",  # 保存策略为steps
        save_steps=50,  # 每50步保存一次
        logging_steps=1,    # logging
        logging_strategy="steps",
        warmup_ratio=0.03,
        learning_rate=2e-5,
        bf16=True,  # 需要与deepspeed配置文件保持一致
        lr_scheduler_type="cosine", # Default is `linear`
        max_length=4096,    # 样本序列最大长度
        label_names=["labels"],
        report_to = "tensorboard",  # show tensorboard
    )

    trainer = SFTTrainer(
        model=base_model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_arguments,
        processing_class=tokenizer, # tokenizer
        peft_config=peft_config,    # lora config
    )

    trainer.train()

if __name__ == "__main__":
    main()
```

保存为`train_sft_lora_qwen2.py`文件，并结合`accelerate`启动并行训练：
```bash
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" accelerate launch --config_file ds_zero1_config.yaml train_sft_lora_qwen2.py
```
> `CUDA_VISIBLE_DEVICES` 用于指定 CUDA 应用程序可见的 GPU 设备。如果需要调整该参数，请同时修改 deepspeed 配置文件中的 `num_processes` 参数，使两者保持一致。

更多配置和使用方式请参考 [GitHub 仓库](https://github.com/JohnWillian/trl_sfttrainer_tutorial.git)。

---

如果觉得这篇文章有用，就给个**赞**👍和**收藏**⭐️吧！也欢迎在评论区分享你的看法！

---

# 9. 参考
- http://huggingface.co/docs/trl/en/sft_trainer
- https://huggingface.co/docs/trl/en/sft_trainer#train-on-assistant-messages-only
- https://github.com/huggingface/transformers/issues/34172
from transformers import AutoTokenizer
from peft import LoraConfig
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

base_model = "./models/Qwen/Qwen2.5-0.5B-Instruct"  # 基座模型路径，自行下载所需模型
train_dataset_name = "./data/train" # Note: 这里是一个目录，不是文件。目录下存放训练数据，例如`train.json`
val_dataset_name = "./data/eval"    # Note: 这里是一个目录，不是文件。目录下存放验证数据，例如`eval.json`
new_model = "../output/finetune/qwen25_05b_sft_lora_r8_a16"   # 训练好的模型保存路径

# SFTConfig中，如果需要使用assistant_only_loss，则需要prompt_template中包含{% generation %}标签
# 但是原版Qwen2.5是不包含的，需手动修改
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
    train_dataset = train_dataset.shuffle(seed=42)  # Shuffle 数据

    # 读取eval数据
    val_dataset = load_dataset(val_dataset_name, split="test")
    val_dataset = val_dataset.shuffle(seed=42)  # Shuffle 数据

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


# 使用 accelerate 启动 deepspeed 训练
# deepspeed配置文件: ds_zero1_config.yaml
# CUDA_VISIBLE_DEVICES 数量应与配置文件中gpu使用数量保持一致
# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" 
# nohup accelerate launch --config_file ds_zero1_config.yaml train_sft_lora_qwen2.py > ../logs/qwen25_05b_sft_lora_r8_a16_training.log 2>&1 &
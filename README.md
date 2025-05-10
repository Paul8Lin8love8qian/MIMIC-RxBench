# MIMIC-RxBench
This dataset contains medical records with injected medication errors, generated using a multi-phase approach involving Large Language Models (LLMs). The dataset is designed for research and development in clinical NLP, error detection, and patient safety. You can see our paper in ...

## Overview

- **Purpose**: To provide annotated examples of common medication errors found in discharge prescriptions.
- **Data Source**: MIMIC-IV-Note: Deidentified free-text clinical notes
- **Error Types**: 7 types of medication errors are included
    - Inappropriate Indication
    - Failure to Consider Allergy
    - Failure to Consider Medical History
    - Inappropriate Dosage Form
    - Drug-Drug Interactions
    - Dosage Inconsistency
    - Incorrect Administration Route

## Environment

## Dataset
You can run the following script to generate data:
```bash
python medical_error_script.py \
  --input_csv /path/to/your/data.csv \
  --output_dir ./output_errors \
  --checkpoint_file ./checkpoint.json \
  --max_records_per_file 100 \
  --model_path "/home/data/models/Qwen2.5-14B-Instruct_1M" \
  --validation_model_path "/home/data/models/Qwen2.5-14B-Instruct_1M" \
  --base_url "http://localhost:8000/v1" \
  --api_key "fake-key"
```

The final dataset reviewed by human can download from <>

## Evaluate
You can run the following script to evaluate MIMIC-RxBench. You should deploy LLM by VLLM or other framework. 

```python
python main.py \
    --dataset MIMIC-RxBench_path \
    --model_endpoint http://127.0.0.1:8058/v1 \
    --model_name model_name \
    --methods prompt_methods
```

The supported elevation engineering methods are as follows.
- **Simple**: Setting basic background and roles, then using straightforward prompts without guiding complex reasoning.
- **COT**: Manually predefining reasoning steps within the prompt templates, guiding the model to follow a fixed logical chain to improve performance on
complex tasks.
- **COT-SC**: Based on CoT, requiring the model to generate multiple independent reasoning paths and determining the final answer via a voting mechanism, thereby enhancing reasoning robustness.
- **MedPrompt**: Combining Few-Shot learning (noting that, due to input
length constraints, the practical setting here was 0-shot) and CoT-SC techniques, prompting the model to dynamically generate a reasoning chain for each specific question and derive the final answer based on the self-generated steps. It is important to note that the Chain-of-Thought used in MedPrompt fundamentally differs from that in the basic methods: the basic CoT adopts a task-level approach where reasoning steps are predefined for the entire task, whereas in MedPrompt, CoT is instance-level, allowing the model to autonomously generate reasoning processes for each individual question, offering greater flexibility and customization.
- **SPP**: Maximizing reasoning and answering capabilities through single-round high-efficiency prompting, involving multi-role playing and collaborative decision-making
- **Self-Refine**: Enabling the model to self-examine and refine its initial answers, progressively improving output quality
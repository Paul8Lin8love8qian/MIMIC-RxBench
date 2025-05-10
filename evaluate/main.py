import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

import argparse
from datetime import datetime
from typing import List, Dict, Tuple
from langchain_openai import ChatOpenAI
from langchain_community.llms import VLLM
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import numpy as np
from tqdm import tqdm
from executors import (
    PromptExecutor,
    SimplePromptExecutor,
    TaskCoTPromptExecutor,
    TaskCoTSelfConsistencyExecutor,
    MedPromptExecutor,
    SPPPromptExecutor,
    SelfRefinePromptExecutor
)
import time
import pdb


class ModelEvaluator:
    def __init__(self, llm, prompt_executor, prompt_name: str,
                 dataset: List[Dict], model_dir: str, model_name: str, batch_size: int=32):
        self.llm = llm
        self.prompt_executor = prompt_executor
        self.prompt_name = prompt_name
        self.dataset = dataset
        self.result = {}
        self.model_dir = model_dir
        self.model_name = model_name
        self.batch_size = batch_size
        self.prediction_file = os.path.join(model_dir, f"predictions_{prompt_name}.json")
        self.process_file = os.path.join(model_dir, f"process_{model_name}_{prompt_name}.json")
        self.predictions = self._load_predictions()
        self.process_results = self._load_process_results()
        self.error_types = ["No Error", "Inappropriate Indication", "Failure to Consider Allergy", 
                           "Failure to Consider Medical History", "Inappropriate Dosage Form", 
                           "Drug-Drug Interactions", "Dosage Inconsistency", "Incorrect Administration Route"]

    def _load_predictions(self) -> List[str]:
        if os.path.exists(self.prediction_file):
            with open(self.prediction_file, 'r', encoding='utf-8') as f:
                predictions = json.load(f)
            print(f"Loaded {len(predictions)} predictions from {self.prediction_file}")
            return predictions
        return [None] * len(self.dataset)

    def _load_process_results(self) -> List[Dict]:
        if os.path.exists(self.process_file):
            with open(self.process_file, 'r', encoding='utf-8') as f:
                process_results = json.load(f)
            print(f"Loaded {len(process_results)} process results from {self.process_file}")
            return process_results
        return [None] * len(self.dataset)

    def _save_predictions(self):
        with open(self.prediction_file, 'w', encoding='utf-8') as f:
            json.dump(self.predictions, f, ensure_ascii=False)

    def _save_process_results(self):
        with open(self.process_file, 'w', encoding='utf-8') as f:
            json.dump(self.process_results, f, ensure_ascii=False, indent=4)

    def predict_batch(self, data_batch: List[Dict]) -> List[tuple[str, Dict]]:
        return self.prompt_executor.generate(data_batch)

    def evaluate(self) -> Dict:
        start_time = time.time()
        y_true = [data["error type"] for data in self.dataset]
        y_pred = []

        total_batches = (len(self.dataset) + self.batch_size - 1) // self.batch_size

        with tqdm(total=total_batches, desc=f"Evaluating {self.prompt_name}", unit="batch") as pbar:
            for start_idx in range(0, len(self.dataset), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(self.dataset))
                batch_data = self.dataset[start_idx:end_idx]
                batch_indices = list(range(start_idx, end_idx))

                # check cached
                cached_predictions = [self.predictions[idx] for idx in batch_indices]
                if all(p is not None for p in cached_predictions):
                    y_pred.extend(cached_predictions)
                    print(f"Batch {start_idx + 1}-{end_idx}/{len(self.dataset)}: Using cached predictions")
                    continue

                # pdb.set_trace()
                batch_results = self.predict_batch(batch_data)
                for idx, (prediction, response) in zip(batch_indices, batch_results):
                    y_pred.append(prediction)
                    self.predictions[idx] = prediction
                    self.process_results[idx] = response
                    # print(f"Sample {idx + 1}/{len(self.dataset)}: Predicted '{prediction}'")

                self._save_predictions()
                self._save_process_results()
                pbar.update(1)  

        overall_metrics = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division="warn")
        per_class_metrics = precision_recall_fscore_support(y_true, y_pred, labels=self.error_types, zero_division="warn")
        conf_matrix = confusion_matrix(y_true, y_pred, labels=self.error_types)

        end_time = time.time()  
        evaluation_time = end_time - start_time  

        self.result = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "prompt_method": self.prompt_name,
            "evaluation_time": evaluation_time/60,
            "overall": {
                "accuracy": float(np.mean(np.array(y_true) == np.array(y_pred))),
                "precision": float(overall_metrics[0]),
                "recall": float(overall_metrics[1]),
                "f1_score": float(overall_metrics[2])
            },
            "per_class": {
                error_type: {
                    "precision": float(per_class_metrics[0][idx]),
                    "recall": float(per_class_metrics[1][idx]),
                    "f1_score": float(per_class_metrics[2][idx]),
                    "support": int(per_class_metrics[3][idx])
                } for idx, error_type in enumerate(self.error_types)
            },
            "confusion_matrix": {
                error_type: conf_matrix[idx].tolist()
                for idx, error_type in enumerate(self.error_types)
            }
        }
        return self.result

    def save_results(self, filename: str):
        with open(os.path.join(self.model_dir, filename), 'w', encoding='utf-8') as f:
            json.dump(self.result, f, ensure_ascii=False, indent=4)

def load_dataset(file_path: str) -> List[Dict]:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def evaluate_all_prompts(llm, dataset: List[Dict], model_name: str, methods: List[str] = None):
    all_results = []
    model_dir = f"results_{model_name}"
    os.makedirs(model_dir, exist_ok=True)

    available_executors = {
        "simple": SimplePromptExecutor(llm),
        "task_cot": TaskCoTPromptExecutor(llm),
        "task_cot_self_consistency": TaskCoTSelfConsistencyExecutor(llm, cache_dir=os.path.join(model_dir, "self_consistency_cache")),
        "medprompt": MedPromptExecutor(llm, cache_dir=os.path.join(model_dir, "medprompt_cache")),
        "spp": SPPPromptExecutor(llm),
        "self_refine": SelfRefinePromptExecutor(llm)
    }

    if methods is None or not methods:
        methods = list(available_executors.keys())

    executors = {name: executor for name, executor in available_executors.items() if name in methods}
    if not executors:
        raise ValueError(f"No valid methods selected. Available methods: {list(available_executors.keys())}")

    for prompt_name, executor in executors.items():
        print(f"\nEvaluating with prompt method: {prompt_name}")
        evaluator = ModelEvaluator(
            llm=llm,
            prompt_executor=executor,
            prompt_name=prompt_name,
            dataset=dataset,
            model_dir=model_dir,
            model_name=model_name,
        )
        
        results = evaluator.evaluate()
        all_results.append(results)
        output_file = f"evaluation_{prompt_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        evaluator.save_results(output_file)
        eval_time = results["evaluation_time"]
        print(f"Evaluation time for {prompt_name}: {eval_time}")
        print(f"Results saved to {os.path.join(model_dir, output_file)}")

    summary_file = f"evaluation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(os.path.join(model_dir, summary_file), 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)
    print(f"Summary saved to {os.path.join(model_dir, summary_file)}")

    print("\nSummary of evaluation times:")
    for result in all_results:
        prompt_name = result["prompt_method"]
        eval_time = result["evaluation_time"]
        print(f"{prompt_name}: {eval_time:.2f} minutes")

    return model_dir, summary_file

def main():
    parser = argparse.ArgumentParser(description="Evaluate a deployed model with multiple prompt methods")
    parser.add_argument("--model_endpoint", default="http://localhost:8000/v1", help="Endpoint URL of the deployed model")
    parser.add_argument("--dataset", required=True, help="Path to the dataset JSON file")
    parser.add_argument("--model_name", default="Qwen2.5-14B-Instruct_1M", help="Name of the model for output files")
    parser.add_argument("--methods", nargs="+", default=None, help="List of prompt or agent methods to evaluate (e.g., simple task_cot)")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    llm = ChatOpenAI(
    model=f"{args.model_name}", 
    base_url=args.model_endpoint, 
    api_key="fake-key", 
    temperature=0.1,
    )
    evaluate_all_prompts(llm, dataset, args.model_name, args.methods)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        sys.exit(1)
from abc import ABC, abstractmethod
from typing import Dict, Tuple, List
from langchain_core.prompts import HumanMessagePromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dataset.utils import MedicationError
import pdb
import os
import json
import hashlib
import re

class PromptExecutor(ABC):
    def __init__(self, llm):
        self.llm = llm
        self.error_types = ["No Error", "Inappropriate Indication", "Failure to Consider Allergy", 
                           "Failure to Consider Medical History", "Inappropriate Dosage Form", 
                           "Drug Interaction", "Dosage Inconsistency", "Incorrect Administration Route"]
        
    def _get_cache_file(self, data: Dict) -> str:
        """Generate a unique cache file name for each input data"""
        data_str = json.dumps(data, sort_keys=True)
        data_hash = hashlib.md5(data_str.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"self_consistency_{data_hash}.json")

    def _load_cache(self, data: Dict) -> List[Dict]:
        """Load cached intermediate results"""
        cache_file = self._get_cache_file(data)
        if os.path.exists(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    
    def _save_cache(self, data: Dict, results: List[Dict]):
        """Save intermediate results to cache"""
        cache_file = self._get_cache_file(data)
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2) 

    def _my_parse_json_text(self, input_text):
        # Regular expression: only extract "choice"
        pattern = r'\s*{\s*"choice"\s*:\s*"([^"]+)"'
        match = re.search(pattern, input_text, re.DOTALL)  
        if match:
            choice = match.group(1)
            process = input_text.strip()
            return {"choice": choice, "process": process}
        else:
            raise ValueError(f"self parase error：{input_text}")

    @abstractmethod
    def generate(self, data_batch: List[Dict], all_error_types_and_explanations: str) -> List[Tuple[str, Dict]]:
        """Generate predictions and processes for a batch of data."""
        pass

    # Extract the content after </think> or "Final Response" in the text
    def extract_after_think(self, text):
        if ("</think>" not in text) and ("Final Response" not in text):
            return text.strip()  # Return original content if neither tag is present
        try:
            # Find the </think> position
            if "</think>" in text:
                think_end = text.index("</think>") + len("</think>")
            if "Final Response" in text:
               think_end = text.index("Final Response") + len("Final Response")
            # Extract the content after </think> or "Final Response"
            json_content = text[think_end:].strip()
            return json_content
        except ValueError:
            return text.strip()# Return original content if the tag is not found

    def _invoke_chain(self, prompt_template, parser, inputs: List[Dict], all_error_types_and_explanations: str, max_retries: int = 5) -> List[Dict]:
        """Helper method to invoke the LLM chain with batch inputs."""
        prompts = [
            [prompt_template.format(
                clinical_note=data["clinical note"],
                discharge_medications=data["discharge medications"],
                all_error_types_and_explanations=all_error_types_and_explanations,
                format_instructions=parser.get_format_instructions()
            )] for data in inputs
        ]
        # pdb.set_trace()
        responses = [self.llm.invoke(prompt) for prompt in prompts]
        results = []
        for i, response in enumerate(responses):
            attempt = 0
            while attempt < max_retries:
                try:
                    # Extract the content after </think> or "Final Response"
                    try:
                        content = self.extract_after_think(response.content)
                        parsed_result = parser.parse(content)
                    except Exception as e:
                        parsed_result = self._my_parse_json_text(content)
                    results.append(parsed_result)
                    break
                except Exception as e:
                    attempt += 1
                    if attempt == max_retries:
                        raise ValueError(f"Parsing error: {e}")
                        # results.append({"choice": "No Error", "process": f"Failed to parse JSON after {max_retries} attempts"})
                    else:
                        single_prompt = prompts[i]
                        responses = [self.llm.invoke(prompt) for prompt in [single_prompt]][0]
        return results
    
    def _get_all_error_types_and_explanations(self) -> List:
        all_error_types_and_explanations = []
        all_medication_error = MedicationError()
        No_Error_explanation = """The prescribed medication aligns with the patient's clinical condition, meets the treatment needs of the current diagnosis or past medical history, and adheres to evidence-based medical principles and regulatory-approved indications. No significant errors are found in dosage form, route of administration, dosage selection, drug interactions, or patient allergy history."""
        for error_type in self.error_types:
            if error_type != "No Error":
                error_type_explanation = all_medication_error.get_error_explanation(error_type)
                all_error_types_and_explanations.append(f"{error_type}: {error_type_explanation}")
            else:
                all_error_types_and_explanations.append(f"No Error: {No_Error_explanation}")

        return all_error_types_and_explanations
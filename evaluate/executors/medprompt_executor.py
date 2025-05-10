import random
from .base_executor import PromptExecutor
from langchain_core.prompts import HumanMessagePromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List, Tuple, Dict
import os 

class MedPromptOutput(BaseModel):
    choice: str = Field(description='Choose one from ["No Error", "Inappropriate Indication", "Failure to Consider Allergy", "Failure to Consider Medical History", "Inappropriate Dosage Form", "Drug Interaction", "Dosage Inconsistency", "Incorrect Administration Route"]')
    process: str = Field(description='Chain-of-thought process')

class MedPromptExecutor(PromptExecutor):
    def __init__(self, llm, cache_dir: str = "self_consistency_cache"):
        super().__init__(llm)
        """MedPrompt-like methods: self-generated chain-of-thought + choice shuffling ensemble (single inference for multiple generations)"""
        self.parser = JsonOutputParser(pydantic_object=MedPromptOutput)
        self.all_error_types_and_explanations = self._get_all_error_types_and_explanations()
        self.prompt_template = HumanMessagePromptTemplate.from_template(
            """You are an experienced inpatient physician. Your task is to determine whether there is a medication error in the discharge medications based on the patient's clinical information using a self-generated chain-of-thought approach. If an error is present, identify its specific type (only one error will exist if applicable).  

            ### **Question**  
            Given the following patient clinical information and discharge medications, is there a medication error, and if so, what type?  
            - **Patient Clinical Information**: 
            {clinical_note}  

            - **Discharge Medications**: 
            {discharge_medications}

            ### **Available Error Types**  
            {all_error_types_and_explanations}

            ### **Answer**  
            Provide a chain-of-thought explanation in the "process" field. Base your explanation on evidence-based medical principles. 
            Then, specify the final answer in the "choice" field.  

            {format_instructions}"""
        )
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True) 

    def generate(self, data_batch: List[Dict]) -> List[Tuple[str, Dict]]:
        max_attempts = 5
        num_samples = 5
        batch_results = []

        for data in data_batch:
            sample_attempts = self._load_cache(data)
            remaining_attempts = num_samples - len(sample_attempts)

            if remaining_attempts > 0:
                for _ in range(remaining_attempts):
                    attempt = 0
                    shuffled_errors = self.all_error_types_and_explanations.copy()
                    random.shuffle(shuffled_errors)
                    shuffled_errors_str = ", ".join(shuffled_errors)
                    
                    response = self._invoke_chain(
                        self.prompt_template, 
                        self.parser, [data], 
                        shuffled_errors_str
                    )[0]

                    while attempt < max_attempts:
                        try:
                            response = self._invoke_chain(
                                self.prompt_template, 
                                self.parser, 
                                [data], 
                                shuffled_errors_str
                            )[0]
                            if response["choice"] in self.error_types:
                                sample_attempts.append({"choice": response["choice"], "process": response["process"]})
                                self._save_cache(data, sample_attempts)
                                break
                            print(f"Attempt {attempt + 1}: Invalid prediction '{response['choice']}' for sample, retrying...")
                            attempt += 1
                        except Exception as e:
                            print(f"Prediction error on attempt {attempt + 1}: {e}")
                            attempt += 1
                    if attempt >= max_attempts:
                        raise RuntimeError(f"Max attempts reached for sample")

            if not sample_attempts:
                raise RuntimeError(f"No valid predictions generated for sample")
            
            choices = [r["choice"] for r in sample_attempts]
            process = "\n".join([f"Attempt {i+1}: {r['process']}" for i, r in enumerate(sample_attempts)])
            final_choice = max(set(choices), key=choices.count)
            final_response = {
                "choice": final_choice,
                "process": f"{process}\nConsistency check: {choices} -> Selected '{final_choice}'",
            }
            batch_results.append((final_choice, final_response))

        return batch_results
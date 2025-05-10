from .base_executor import PromptExecutor
from langchain_core.prompts import HumanMessagePromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List, Tuple, Dict

class TaskCoTOutput(BaseModel):
    choice: str = Field(description='Choose one from ["No Error", "Inappropriate Indication", "Failure to Consider Allergy", "Failure to Consider Medical History", "Inappropriate Dosage Form", "Drug Interaction", "Dosage Inconsistency", "Incorrect Administration Route"]')
    process: str = Field(description='Step-by-step reasoning following the CoT steps')

class TaskCoTPromptExecutor(PromptExecutor):
    def __init__(self, llm):
        super().__init__(llm)
        self.parser = JsonOutputParser(pydantic_object=TaskCoTOutput)
        self.all_error_types_and_explanations = self._get_all_error_types_and_explanations()
        self.prompt_template = HumanMessagePromptTemplate.from_template(
            """You are an experienced inpatient physician. Your task is to determine whether there is a medication error in the discharge medications based on the patient's clinical information using a step-by-step reasoning process. If an error is present, identify its specific type (only one error will exist if applicable). 

            ### **Task Requirements**  
            1. **Step-by-Step Reasoning**: Follow these steps to analyze the problem and include the reasoning in your output:  
               - Step 1: Review the patient's clinical information and identify key conditions, allergies, and medical history.  
               - Step 2: Examine the discharge medications and check for appropriateness based on Step 1.  
               - Step 3: Compare findings against the available error types to determine if an error exists.  
               - Step 4: If an error exists, select the most appropriate error type; otherwise, conclude 'No Error'. 
            2. **Ensure medical accuracy**: Base your reasoning on evidence-based medical principles. 
            3. **Strict JSON Output**: Provide the result in JSON format, including both the reasoning process and the final choice. The reasoning should be detailed in the "process" field, and the final answer in the "choice" field.

            ### **Input Information**  
            - **Patient Clinical Information**: 
            {clinical_note}  

            - **Discharge Medications**: 
            {discharge_medications}  

            - **Available Error Types**: {all_error_types_and_explanations}  

            {format_instructions}"""
        )

    def generate(self, data_batch: List[Dict]) -> List[Tuple[str, Dict]]:
        max_attempts = 5
        results = []

        try:
            responses = self._invoke_chain(self.prompt_template, self.parser, data_batch, ", ".join(self.all_error_types_and_explanations))
            batch_results = []
            all_valid = True
            for idx, response in enumerate(responses):
                if response["choice"] in self.error_types:
                    batch_results.append((response["choice"], response))
                else:
                    all_valid = False
                    break
            if all_valid:
                return batch_results
            raise ValueError("Invalid choices in the batch results")
        except Exception as e:
            raise e
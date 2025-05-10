from .base_executor import PromptExecutor
from langchain_core.prompts import HumanMessagePromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List, Tuple, Dict

class SimpleOutput(BaseModel):
    choice: str = Field(description='Choose one from ["No Error", "Inappropriate Indication", "Failure to Consider Allergy", "Failure to Consider Medical History", "Inappropriate Dosage Form", "Drug Interaction", "Dosage Inconsistency", "Incorrect Administration Route"]')
    process: str = Field(description='Brief reasoning for the choice')

class SimplePromptExecutor(PromptExecutor):
    def __init__(self, llm):
        super().__init__(llm)
        self.parser = JsonOutputParser(pydantic_object=SimpleOutput)
        self.all_error_types_and_explanations = self._get_all_error_types_and_explanations()
        self.prompt_template = HumanMessagePromptTemplate.from_template(
            """You are an experienced inpatient physician. Your task is to determine whether there is a medication error in the discharge medications based on the patient's clinical information. If an error is present, identify its specific type (only one error will exist if applicable).  

            ### **Task Requirements**  
            1. **Determine the presence of a medication error**: Analyze whether the discharge medications are appropriate based on the patient's clinical condition, medical history, and other relevant information. 
            2. **Identify the error type (if applicable)**: If a medication error is detected, select the most appropriate category from the given error types (only one error will be present).  
            3. **Ensure medical accuracy**: Follow evidence-based medical principles to assess medication appropriateness without making unfounded assumptions.  
            4. **Provide a clear output format**: Clearly state 'No Error' if no issues are found. If an error is identified, specify the corresponding error type.  
            5. **Strict JSON Output**: Provide the result in JSON format without additional explanation outside the process field. 

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
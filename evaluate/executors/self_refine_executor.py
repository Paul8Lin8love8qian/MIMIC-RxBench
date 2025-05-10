from .base_executor import PromptExecutor
from langchain_core.prompts import HumanMessagePromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List, Tuple, Dict

class SimpleOutput(BaseModel):
    choice: str = Field(description='Choose one from ["No Error", "Inappropriate Indication", "Failure to Consider Allergy", "Failure to Consider Medical History", "Inappropriate Dosage Form", "Drug Interaction", "Dosage Inconsistency", "Incorrect Administration Route"]')
    process: str = Field(description='Brief reasoning for the choice')

class FeedbackOutput(BaseModel):
    status: str = Field(description='"satisfactory" if the choice is reasonable, "needs refinement" if unreasonable')
    feedback: str = Field(description='Detailed explanation of your judgment and a suggested error type if applicable')

class SelfRefineOutput(BaseModel):
    choice: str = Field(description='Choose one from ["No Error", "Inappropriate Indication", "Failure to Consider Allergy", "Failure to Consider Medical History", "Inappropriate Dosage Form", "Drug Interaction", "Dosage Inconsistency", "Incorrect Administration Route"]')
    process: str = Field(description='Detailed reasoning for the refined choice')

class SelfRefinePromptExecutor(PromptExecutor):
    def __init__(self, llm):
        super().__init__(llm)
        self.all_error_types_and_explanations = self._get_all_error_types_and_explanations()
        # Initial generation prompt (pgen)
        self.pgen_template = HumanMessagePromptTemplate.from_template(
            """You are an experienced inpatient physician. Generate an initial assessment to determine if there is a medication error in the discharge medications based on the patient's clinical information.If an error is present, identify its specific type (only one error will exist if applicable). Provide your choice and reasoning.
            - Clinical Note: 
            {clinical_note}

            - Discharge Medications: 
            {discharge_medications}

            - Available Error Types: 
            {all_error_types_and_explanations}

            {format_instructions}"""
        )

        # Feedback prompt (pfb)
        self.pfb_template = HumanMessagePromptTemplate.from_template(
            """You are a critical reviewer. Your task is to evaluate whether the current error type choice in the assessment is reasonable based on the patient's clinical information and discharge medications. Follow these steps:

            1. **Evaluate the Current Choice**: Review the 'choice' in the current assessment and determine if it accurately reflects the medication error (or lack thereof) given the clinical note, discharge medications, and available error types.
            2. **Provide Feedback**:
                - If the choice is reasonable, set 'status' to 'satisfactory' and explain why it correctly identifies the error type (or lack of error).
                - If the choice is unreasonable, set 'status' to 'needs refinement', explain why it is incorrect, and suggest a more appropriate error type from the following list: {error_types}.
                
            - Clinical Note: 
            {clinical_note}

            - Discharge Medications: 
            {discharge_medications}

            - Current Assessment: 
            {current_output}

            - Available Error Types with Explanations: {all_error_types_and_explanations}

            {format_instructions}"""
        )

        # Refinement prompt (prefine)
        self.prefine_template = HumanMessagePromptTemplate.from_template(
            """You are an experienced inpatient physician. Refine the assessment by adjusting the error type choice based on the feedback provided. Follow these steps:
    
            1. **Review Feedback**: Analyze the feedback, focusing on why the previous error type choice was deemed unreasonable and the suggested alternative.
            2. **Adjust the Error Type**: Based on the clinical note, discharge medications, initial output, and feedback, select a more appropriate error type from the available options.
            3. **Provide Reasoning**: Explain why the new choice better reflects the situation.

            - Clinical Note: 
            {clinical_note}

            - Discharge Medications: 
            {discharge_medications}

            - Previous Outputs and Feedback: 
            {history}

            - Available Error Types: 
            {all_error_types_and_explanations}
            
            {format_instructions}"""
        )

        self.simple_parser = JsonOutputParser(pydantic_object=SimpleOutput)
        self.feedback_parser = JsonOutputParser(pydantic_object=FeedbackOutput)
        self.selfrefine_parser = JsonOutputParser(pydantic_object=SelfRefineOutput)

    def _invoke_chain(self, prompt_template, parser, data_batch: List[Dict], max_retries: int = 3, **kwargs) -> List[Dict]:
        """Custom invoke_chain for Self-Refine, handling different prompt formats."""
        prompts = []
        for data in data_batch:
            if prompt_template == self.pgen_template:
                prompt = prompt_template.format(
                    clinical_note=data["clinical note"],
                    discharge_medications=data["discharge medications"],
                    all_error_types_and_explanations=", ".join(self.all_error_types_and_explanations),
                    format_instructions=parser.get_format_instructions()
                )
            elif prompt_template == self.pfb_template:
                prompt = prompt_template.format(
                    clinical_note=data["clinical note"],
                    discharge_medications=data["discharge medications"],
                    current_output=kwargs.get("current_output", [{}])[data_batch.index(data)],
                    all_error_types_and_explanations=", ".join(self.all_error_types_and_explanations),
                    error_types=", ".join(self.error_types),
                    format_instructions=parser.get_format_instructions()
                )
            elif prompt_template == self.prefine_template:
                prompt = prompt_template.format(
                    clinical_note=data["clinical note"],
                    discharge_medications=data["discharge medications"],
                    initial_output=kwargs.get("initial_output", [{}])[data_batch.index(data)],
                    history=kwargs.get("history", [""])[data_batch.index(data)],
                    all_error_types_and_explanations=", ".join(self.all_error_types_and_explanations),
                    format_instructions=parser.get_format_instructions()
                )
            prompts.append([prompt])


        responses = [self.llm.invoke(prompt) for prompt in prompts]

        # Process each response individually with retries
        results = []
        for i, response in enumerate(responses):
            attempt = 0
            while attempt < max_retries:
                try:
                    parsed_result = parser.parse(self.extract_after_think(response.content))
                    results.append(parsed_result)
                    break
                except Exception as e:
                    attempt += 1
                    if attempt == max_retries:
                        # If all retries fail, return "unknown" for this sample only
                        if prompt_template == self.pfb_template:
                            results.append({"status": "needs_refinement", "feedback": "Failed to parse feedback after multiple attempts"})
                        else:  # pgen or prefine
                            raise e

                    else:
                        # Retry by re-invoking for this single sample
                        single_prompt = prompts[i]
                        responses = [self.llm.invoke(prompt) for prompt in [single_prompt]][0]
                        # response = self.llm.batch([single_prompt])[0]

        return results


    def stop_condition(self, feedback: Dict, iteration: int) -> bool:
        """Stop if status is satisfactory or max iterations reached."""
        max_iterations = 3
        if iteration >= max_iterations:
            return True
        return feedback["status"] == "satisfactory"

    def generate(self, data_batch: List[Dict]) -> List[Tuple[str, Dict]]:
        batch_results = []

        # Step 1: Initial generation 
        y0_batch = self._invoke_chain(
            self.pgen_template, 
            self.simple_parser, 
            data_batch, 
        )

        # Step 2: Process each sample individually for refinement
        for i, data in enumerate(data_batch):
            current_output = y0_batch[i]
            history = [f"Initial Output: {current_output['choice']} - {current_output['process']}"]

            # Skip refinement if initial output is "unknown"
            if current_output["choice"] == "unknown":
                final_response = SelfRefineOutput(
                    choice="unknown",
                    process="\n".join(history),
                )
                batch_results.append(("unknown", final_response.dict()))
                continue

            # Iterative refinement for this sample
            for t in range(10):  # Max 10 iterations as a safeguard
                # Step : Feedback
                fb_t = self._invoke_chain(
                    self.pfb_template, 
                    self.feedback_parser, 
                    [data], 
                    current_output=[str(current_output)]
                )[0]

                history.append(f"Iteration {t}: Feedback - {fb_t['feedback']} (Status: {fb_t['status']})")

                # Step : Check stop condition
                if self.stop_condition(fb_t, t):
                    break

                # Step : Refine
                next_output = self._invoke_chain(
                    self.prefine_template, 
                    self.simple_parser, 
                    [data], 
                    initial_output=[str(y0_batch[i])],
                    history=["\n".join(history)]
                )[0]

                current_output = next_output
                history.append(f"Refined Output: {current_output['choice']} - {current_output['process']}")

            # Step : Construct final result for this sample
            final_response = {
                "choice": current_output["choice"],
                "process": "\n".join(history),
            }
            batch_results.append((current_output["choice"], final_response))

        return batch_results
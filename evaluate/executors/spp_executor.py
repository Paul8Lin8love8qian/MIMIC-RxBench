from .base_executor import PromptExecutor
from langchain_core.prompts import HumanMessagePromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List, Tuple, Dict

class SPPOutput(BaseModel):
    choice: str = Field(description='Final consensus choice')
    process: str = Field(description='Detailed collaboration process among participants')

class SPPPromptExecutor(PromptExecutor):
    """SPPPromptExecutor: Judging Medication Errors through Multi-role Self-collaboration, Retaining Original Examples from the Paper"""
    def __init__(self, llm):
        super().__init__(llm)
        self.parser = JsonOutputParser(pydantic_object=SPPOutput)
        self.all_error_types_and_explanations = self._get_all_error_types_and_explanations()
        self.prompt_template = HumanMessagePromptTemplate.from_template(
            '''When faced with a task, begin by identifying the participants who will contribute to solving the task. Then, initiate a multi-round collaboration process until a final solution is reached. The participants will give critical comments and detailed suggestions whenever necessary.

            Here are some examples:
            ---
            Example Task 1: Use numbers and basic arithmetic operations (+ - * /) to obtain 24. You need to use all numbers, and each number can only be used once.
            Input: 6 12 1 1

            Participants: AI Assistant (you); Math Expert

            Start collaboration!

            Math Expert: Let's analyze the task in detail. You need to make sure that you meet the requirement, that you need to use exactly the four numbers (6 12 1 1) to construct 24. To reach 24, you can think of the common divisors of 24 such as 4, 6, 8, 3 and try to construct these first. Also you need to think of potential additions that can reach 24, such as 12 + 12.
            AI Assistant (you): Thanks for the hints! Here's one initial solution: (12 / (1 + 1)) * 6 = 24
            Math Expert: Let's check the answer step by step. (1+1) = 2, (12 / 2) = 6, 6 * 6 = 36 which is not 24! The answer is not correct. Can you fix this by considering other combinations? Please do not make similar mistakes.
            AI Assistant (you): Thanks for pointing out the mistake. Here is a revised solution considering 24 can also be reached by 3 * 8: (6 + 1 + 1) * (12 / 4) = 24.
            Math Expert: Let's first check if the calculation is correct. (6 + 1 + 1) = 8, 12 / 4 = 3, 8 * 3 = 24. The calculation is correct, but you used 6 1 1 12 4 which is not the same as the input 6 12 1 1. Can you avoid using a number that is not part of the input?
            AI Assistant (you): You are right, here is a revised solution considering 24 can be reached by 12 + 12 and without using any additional numbers: 6 * (1 - 1) + 12 = 24.
            Math Expert: Let's check the answer again. 1 - 1 = 0, 6 * 0 = 0, 0 + 12 = 12. I believe you are very close, here is a hint: try to change the '1 - 1' to '1 + 1'.
            AI Assistant (you): Sure, here is the corrected answer:  6 * (1+1) + 12 = 24
            Math Expert: Let's verify the solution. 1 + 1 = 2, 6 * 2 = 12, 12 + 12 = 12. You used 1 1 6 12 which is identical to the input 6 12 1 1. Everything looks good!

            Finish collaboration!

            Final answer: 6 * (1 + 1) + 12 = 24

            ---
            Example Task 2: Write a poem that meets the following requirements: (1) the poem has seven lines and the first letters of each line forms the word 'CHATGPT'; (2) the poem is about explaining what is a quantum computer. (3) the poem needs to be easy to understand by a ten years old kid.

            Participants: AI Assistant (you); Poet; Computer Scientist; Ten year old child

            Poet: Make sure that you write the poem with seven lines, and the first letters of the lines should be C, H, A, T, G, P, T.
            Computer Scientist: A quantum computer is an advanced computing device that uses the principles of quantum mechanics to process and store information. Unlike classical computers that use bits to represent information as 0s and 1s, quantum computers use quantum bits or qubits. Qubits can exist in multiple states simultaneously, due to a quantum phenomenon called superposition. You can consider using these information for the poem.
            Ten year old child: I hope the poem to be fun and easy to understanding. I don't want to see a lot of jargons or complicated concepts.
            AI Assistant (you): Thanks for the guidance! Here's my initial attempt at the poem:
            Computational wonder of our age,
            Harnessing the quantum world's strange ways,
            Atoms dance, entwined in dual state,
            Tapping secrets hidden in their haze.

            Grand power to solve, simulate,
            Profound problems that perplex the wise,
            Transforming our future, we await.

            Poet: Let's verify if the poem meets the requirements. The first letters are CHATGPT which is correct! And the poem rhymes well. Good job!
            Computer Scientist: Everything looks good to me!
            Ten year old child: I don't know what does perplex mean. Can you make the use of words easier to understand?
            AI Assistant (you): Sure, let me revise the poem by using more common words. Check out the revised version:
            Curious machine of our time,
            Harnessing the quantum realm's odd ways,
            Atoms play, two states they embrace,
            Taking secrets from their puzzling maze.

            Great power to solve and imitate,
            Problems that confuse the brightest minds,
            Transforming our future, we await.

            Poet: Let's check again if the poem meets the requirements. The first letters are C H A T G P T. And now the poem is more accessible to children. Everything looks good to me.
            Computer Scientist: Looking good!
            Ten year old child: I like this version a lot!

            Finish collaboration!

            Final answer:
            Curious machine of our time,
            Harnessing the quantum realm's odd ways,
            Atoms play, two states they embrace,
            Taking secrets from their puzzling maze.

            Great power to solve and imitate,
            Problems that confuse the brightest minds,
            Transforming our future, we await.

            ---
            Now, identify the participants and collaboratively solve the following task step by step. Remember to present your final solution in strict JSON format.

            Task: Determine whether there is a medication error in the discharge medications based on the patient's clinical information. If an error is present, identify its specific type (only one error will exist if applicable).

            Input:
            - **Patient Clinical Information**: 
            {clinical_note}  

            - **Discharge Medications**: 
            {discharge_medications}  

            - **Available Error Types**: {all_error_types_and_explanations}  

            ### **Requirements**
            1. Identify participants relevant to the task (e.g., AI Assistant, Physician, Pharmacist, etc.).
            2. Each participant contributes critical comments or suggestions based on their expertise.
            3. AI Assistant proposes an initial solution, followed by feedback and revisions until consensus.

            {format_instructions}'''
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
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage, BaseMessage
from langchain_core.prompts import (ChatPromptTemplate,
                                    SystemMessagePromptTemplate, 
                                    HumanMessagePromptTemplate,)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import JsonOutputParser
import pdb
from langgraph.graph import StateGraph, START, END
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, 'utils')))
from tool import openfda_tool
from langgraph.checkpoint.memory import MemorySaver


MP_SYSTEM_TEMPLATE = """You are a rigorous clinical pharmacy expert focused on generating medication prescription data containing a single specified type of error. Your goal is to ensure that the introduced error is precise and accurate, while the prescription context appears clinically plausible but actually contains the designated error.

**Task Description:**  
Based on the following input information, generate a complete prescription, deliberately introducing the specified medication error:  
1. Free-text clinical notes: Including the patient’s clinical background (e.g., medical history, allergy history, indications) and authentic discharge medication prescription information;  
2. Specified medication error type: The designated error category, definition, and available error injection strategies for reference;  
3. Past failed generation cases and reasons (if any): Provided as optimization references to avoid repeating issues.  

**Core Requirements:**  
1. The objective is to generate prescription data containing the specified error, not to pursue medical correctness or optimize the prescription;  
2. The generated prescription must integrate the patient’s clinical background and treatment logic, appearing relevant to the condition while actually containing the target error;  
3. Strictly limit the error type to the specified single error, avoiding interference from other non-target errors;  
4. The generated prescription should maintain the professionalism of clinical terminology, ensuring the data format aligns with real-world prescriptions;  
5. You need to modify the prescription data based on searched drug information rather than relying solely on your own knowledge.
"""

VLP_SYSTEM_TEMPLATE = """You are a professional clinical pharmacist. Your task is to verify the generated medication error data based on the input information, ensuring that the data accurately reflects the target error type and does not introduce other non-specified error types due to modifications, thereby maintaining error focus.  

**Input Information Categories:**  
1. De-identified free-text clinical notes: Provide the patient's clinical background and original discharge medications information.  
2. Specified medication error type and its explanation.  
3. Drug information referenced during generation.  
4. Modified discharge medications.  
5. Modification rationale.
"""

VLP_USER_TEMPLATE = """Please verify the generated medication error data based on the following provided information and provide the verification results:  

**Provided Information:**  
1. Deidentified free-text clinical notes:  
{Deidentified_Free_Text_Clinical_Notes}  

2. Error type and its explanation:  
{Error_Type}: {Error_Type_Explanation}  

3. Drug information referenced during generation:  
{Referenced_Drug_Information}  

4. Modified discharge medications:  
{Modified_Discharge_Medications}  

5. Modification rationale:  
{Modification_Rationale}  

{format_instructions} 
"""

# Medication Error Type, Explanation and Injection Strategy
class MedicationError:
    def __init__(self):
        self.Inappropriate_Indication = "The prescribed medication does not match the patient's current diagnosis or past medical history treatment needs, meaning the drug does not fulfill the patient’s clinical requirements and lacks a legitimate medical purpose or evidence-based support. For example, prescribing insulin to a patient without diabetes or prescribing antibiotics to a patient without evidence of infection."
        self.Inappropriate_Indication_EIS="For example, replacing a drug used to treat a disease related to the discharge diagnosis with a drug for a condition the patient does not have, and using drug information lookup tools to obtain the necessary drug indications and related information."

        self.Failure_to_Consider_Allergy = "The prescribed medication is a known allergen for the patient or carries a risk of cross-reactivity, but no appropriate adjustments or substitutions have been made."
        self.Failure_to_Consider_Allergy_EIS = "For example, searching for drug information and replacing a drug with another that could trigger an allergic reaction, even though the new drug has the same therapeutic effect as the original one, but the patient's allergy history was not fully taken into account."

        self.Failure_to_Consider_Medical_History = "The prescribed medication is appropriate for a certain disease or symptom but does not account for the patient’s past medical history, comorbidities, organ function (e.g., liver or renal function), pregnancy, or lactation status, leading to potential safety or efficacy risks. For example, a patient with a history of asthma admitted to the ICU for septic shock is prescribed a non-selective beta-blocker (e.g., propranolol) upon discharge, which may exacerbate asthma symptoms."
        self.Failure_to_Consider_Medical_History_EIS = "For example, searching for drug information and replacing a drug with another that may pose risks related to the patient's medical history, even though the new drug has the same therapeutic effect as the original one."

        self.Inappropriate_Dosage_Form = "The physical form of the prescribed medication is inappropriately selected, failing to optimize based on the patient’s clinical condition (e.g., urgency of condition, swallowing ability), ease of administration, or drug availability, potentially affecting efficacy, safety, or patient adherence. This type of error pertains to the selection of the drug’s dosage form itself, rather than the implementation of the administration route. Examples include: prescribing a sustained-release formulation when rapid onset is needed; prescribing large, non-crushable tablets for a patient with swallowing difficulties; or prescribing a dosage form that does not exist on the market (e.g., Levemir (insulin detemir) 50 units oral QHS, when insulin is not typically available in an oral form)."
        self.Inappropriate_Dosage_Form_EIS = "For example, searching for drug information and replacing the dosage form of a drug with one that does not match the patient's needs (e.g., replacing an injection with a tablet)."

        self.Drug_Interaction = "The simultaneous use of two or more medications results in a known pharmacological interaction, which may reduce therapeutic efficacy, increase adverse effects, or elevate toxicity risk."
        self.Drug_Interaction_EIS = "For example, searching for drug information and replacing a drug with another that has the same therapeutic effect, but the new drug has a contraindicated interaction with other medications."

        self.Dosage_Inconsistency = "The prescribed dosage, administration frequency, or treatment duration deviates from the recommended regimen approved by drug regulatory authorities, potentially leading to underdosing or overdosing."
        self.Dosage_Inconsistency_EIS = "For example, searching for drug information and modifying the dose of a drug to a non-recommended dose."


        self.Incorrect_Administration_Route = "Refers to a situation where the administration route of a drug is inconsistent with the approved or recommended route for that dosage form, resulting in an inappropriate method of drug delivery into the body. This type of error typically occurs when the dosage form itself is acceptable or explicitly specified, potentially affecting safety, bioavailability, or efficacy due to the incorrect administration route. Examples include: a tablet intended for oral use being erroneously prescribed for intravenous injection; a drug meant for intravenous injection being mistakenly prescribed for intramuscular injection; or an ointment intended for topical use being incorrectly prescribed for oral administration."
        self.Incorrect_Administration_Route_EIS = "For example, searching for drug information and modifying the administration route of a drug to an incorrect one."

        self.medication_errors =  {"Inappropriate Indication": self.Inappropriate_Indication,
                         "Failure to Consider Allergy": self.Failure_to_Consider_Allergy,
                         "Failure to Consider Medical History": self.Failure_to_Consider_Medical_History,
                         "Inappropriate Dosage Form": self.Inappropriate_Dosage_Form,
                         "Drug-Drug Interactions": self.Drug_Interaction,
                         "Dosage Inconsistency": self.Dosage_Inconsistency,
                         "Incorrect Administration Route": self.Incorrect_Administration_Route}
        
        self.injection_strategy = {"Inappropriate Indication": self.Inappropriate_Indication_EIS,
                         "Failure to Consider Allergy": self.Failure_to_Consider_Allergy_EIS,
                         "Failure to Consider Medical History": self.Failure_to_Consider_Medical_History_EIS,
                         "Inappropriate Dosage Form": self.Inappropriate_Dosage_Form_EIS,
                         "Drug-Drug Interactions": self.Drug_Interaction_EIS,
                         "Dosage Inconsistency": self.Dosage_Inconsistency_EIS,
                         "Incorrect Administration Route": self.Incorrect_Administration_Route_EIS}
    
    def get_error_type(self):
        return list(self.medication_errors.keys())
    
    def get_error_explanation(self, error_type):
        return self.medication_errors[error_type]
    
    def get_injection_strategy(self, error_type):
        return self.injection_strategy[error_type]
    
    def get_error_type(self):
        return list(self.medication_errors.keys())
    
    def get_error_explanation(self, error_type):
        return self.medication_errors[error_type]
    
    def get_injection_strategy(self, error_type):
        return self.injection_strategy[error_type]
    
class Modification_outformat(BaseModel):
    modified_discharge_medications: str = Field(description="The complete discharge medications after modification.It should be consistent with the original discharge medications form.")
    modification_rationale: str = Field(description="Explanation of how the modifications fulfill the specified error type and waht specific changes you have made.")

class Validation_outformat(BaseModel):
    verification_result: str = Field(description="Verify whether the modification was successful, true or false.")
    failure_reason: str = Field(description="If the modification fails, provide the reason; if successful, leave empty.")

class State(TypedDict):
    messages: Annotated[list, add_messages]
    tool_retry_count: int
    initial_data: dict
    referenced_drug_information: str
    search_arg: str

class PreliminaryThinkingNode:
    def __init__(self, llm_with_tools):
        self.llm_with_tools = llm_with_tools
    
    def __call__(self, state: State) -> State:
        print(f"Now entering the PreliminaryThinkingNode")

        MP_user_prompt_text = """"Based on the provided free-text clinical notes, deeply understand the patient’s clinical background and prescription logic, extracting information closely related to the specified error type (e.g., indications, medical history, allergy history, dosage, administration route, etc.). Refer to the definition and injection strategies of the error type for preliminary analysis, and clarify the following:  

        1. Drug information to be searched (e.g., drug name, dosage range, contraindications, etc.);  
        2. The basis and objective of the search (e.g., what information to search for and how to ensure it relates to the error type);  
        3. The direction of modification after obtaining the information (e.g., how to adjust the prescription data based on search results to introduce the specified error). Note: This step is only a preliminary concept and does not involve direct modification; specific adjustments should be made after obtaining accurate drug information.  
        4. If past failed generation cases and their reasons are provided, analyze the issues (e.g., inappropriate parameter selection, illogical reasoning, etc.) to avoid repeating similar problems.

        Deidentified free-text clinical notes:
        {Deidentified_Free_Text_Clinical_Notes}
        {Discharge_medications}

        Error type:
        {Error_Type}

        Error type explanation:
        {Error_Type_Explanation}

        Error Injection Strategy：
        {Error_Injection_Strategy}

        Failed Cases and Reasons (if any):
        {Failed_Cases_and_Reasons}
        """
        initial_data = state["initial_data"]
        user_template = HumanMessagePromptTemplate.from_template(MP_user_prompt_text)
        user_message = user_template.format(Deidentified_Free_Text_Clinical_Notes=initial_data["Deidentified_Free_Text_Clinical_Notes"], 
                                            Discharge_medications=initial_data["Discharge_medications"],
                                            Error_Type=initial_data["Error_Type"], Error_Type_Explanation=initial_data["Error_Type_Explanation"], Error_Injection_Strategy=initial_data["Error_Injection_Strategy"], Failed_Cases_and_Reasons=initial_data["Failed_Cases_and_Reasons"])
        # user_message.pretty_print()
        updated_messages = add_messages(state['messages'], [user_message])
        response = self.llm_with_tools.invoke(updated_messages)
        
        return {"messages": [user_message] + [response]}

class MakeQueryParamNode:
    def __init__(self, llm_with_tools):
        self.llm_with_tools = llm_with_tools    

    def __call__(self, state: State) -> State:
        print(f"Now entering the MakeQueryParamNode")
        user_text = """Please use the provided drug information search tool based on your preliminary analysis.  
        - If this is the first use, carefully review the previous thinking and plan to ensure the search parameters accurately reflect the needs, providing support for subsequently generating prescription data containing the error.  
        - If a previous search attempt failed, revisit the analysis and plan, adjusting the search parameters to address the issue. 

        Note that the tool uses AND logic (only returning information that meets all conditions), so select search parameters carefully.
        """
        user_message = HumanMessage(user_text)
        
        # user_message.pretty_print()
        updated_messages = add_messages(state['messages'], [user_message])   
        response = self.llm_with_tools.invoke(updated_messages)

        return {"messages": [user_message] + [response], "search_arg": response.tool_calls[0]["args"]}

def retry_make_param(state: State) -> str:
    messages = state["messages"]
    last_message = messages[-1]
    if isinstance(last_message, AIMessage):
        if last_message.tool_calls:
            return "success"
        else:
            # retry
            state["messages"] = messages[:-2]
            return "retry"
    return "failed"

class FDAToolNode:
    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, state: State) -> State:
        print(f"Now entering the FDAToolNode")
        if messages := state.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            try:
                drug_info = self.tools_by_name[tool_call["name"]].invoke(tool_call["args"])
                outputs.append(
                    ToolMessage(
                        content=drug_info,
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"],
                    )
                )

                return {"messages": outputs, "referenced_drug_information": drug_info}
            
            except Exception as e:
                text = f"{e}.\n Try calling the tool again with other arguments. Do not repeat mistakes."
                print(text, "\nRe-entering the MakeQueryParamNode")
                outputs.append(
                    ToolMessage(
                        content=text,
                        tool_call_id=tool_call["id"],
                        additional_kwargs={'tool_call_error': True}
                    )
                )                
                return {"messages": outputs}


def CheckToolResultNode(state: State) -> str:
    print(f"Now entering the CheckToolResultNode")
    last_message = state["messages"][-1]
    if isinstance(last_message, ToolMessage):
        tool_call_error = last_message.additional_kwargs.get("tool_call_error", False)
        if tool_call_error:
            state["tool_retry_count"] += 1
    return {"tool_retry_count": state["tool_retry_count"]}


def tool_call_retry(state: State) -> str:
    last_message = state["messages"][-1]
    if isinstance(last_message, ToolMessage):
        tool_call_error = last_message.additional_kwargs.get("tool_call_error", False)
        # Check if the tool call returned an error or invalid result (adjust condition based on the tool's response format)
        if tool_call_error:
            if state["tool_retry_count"] > 5:  # Maximum retry count is 5
                return "failed"
            return "retry"  # Return retry, transition to QPNode
        else:
            return "success"  # Return success, proceed to next step
    return "retry"  # If not a ToolMessage, also return retry

class Modify_Node:
    def __init__(self, llm_with_tools):
        self.llm_with_tools = llm_with_tools

    def __call__(self, state: State) -> State:
        print(f"当前处于 Modify_Node 节点")
        messages = state["messages"]
        last_message = messages[-1]
        if not isinstance(last_message, ToolMessage):
            raise ValueError("Expected last message to be a ToolMessage with tool results.")

        parser = JsonOutputParser(pydantic_object=Modification_outformat)

        modification_template_text = """Now, based on the drug information obtained from the search, generate a complete prescription dataset containing the specified medication error, ensuring the output meets the requirements. Note that only one drug should be modified, and the output should follow the prescribed format without any extraneous content.
        
        {format_instructions}
        """
        modification_template = HumanMessagePromptTemplate.from_template(modification_template_text, partial_variables={"format_instructions": parser.get_format_instructions()})
        user_message = modification_template.format()

        #  Invoke the LLM to generate the revised result
        updated_messages = add_messages(messages, [user_message])
        response = self.llm_with_tools.invoke(updated_messages)

        return {"messages": [user_message] + [response]}

def build_graph(llm_with_tools):
    
    graph_builder = StateGraph(State)

    PTNode = PreliminaryThinkingNode(llm_with_tools)

    graph_builder.add_node("PTNode", PTNode)
    graph_builder.add_edge(START, "PTNode")

    QPNode = MakeQueryParamNode(llm_with_tools)
    graph_builder.add_node("QPNode", QPNode)
    graph_builder.add_edge("PTNode", "QPNode")

    FDANode = FDAToolNode(tools=[openfda_tool])
    graph_builder.add_node("FDANode", FDANode)
    graph_builder.add_conditional_edges("QPNode", 
                                        retry_make_param, 
                                        {"retry": "QPNode", 
                                        "success": "FDANode", 
                                        "failed": END})

    # Add a CheckNode and set up conditional transitions
    graph_builder.add_node("CheckNode", CheckToolResultNode)
    graph_builder.add_edge("FDANode", "CheckNode")

    ModifyNode = Modify_Node(llm_with_tools)
    graph_builder.add_node("ModifyNode", ModifyNode)

    # Conditional edges: jump based on the return value of CheckNode
    graph_builder.add_conditional_edges(
        "CheckNode",
        tool_call_retry,
        {
            "retry": "QPNode",  # If the result is empty, go back to QPNode to regenerate parameters
            "success": "ModifyNode",       # If the result is valid, connect to the next node
            "failed": END
        }
    )

    graph_builder.add_edge("ModifyNode", END)

    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory)

    return graph


def Modification_phase(
        llm_with_tools, 
        note: str, 
        discharge_medications: str, 
        error_type: str, 
        error_type_explanation: str, 
        error_injection_strategy: str, 
        failed_cases_and_reasons: str
) -> tuple[str, str, str, str]:
    """
    Modification Phase: Generate a discharge prescription with specified errors.

    Args:
        Mllm_with_tools: The LLM model used in the modification phase (with tools bound).
        note: Clinical note.
        discharge_medications: Original discharge medications.
        error_type: Type of error to be injected.
        error_type_explanation: Explanation of the error type.
        error_injection_strategy: Strategy for injecting errors.
        failed_cases_and_reasons: Record of failed cases and their reasons.

    Returns:
        Tuple: (search_arg, referenced_drug_information, modified_discharge_medications, modification_rationale)
    """   
    
    # Build the workflow
    graph = build_graph(llm_with_tools)

    # Initialize the state information
    system_template = SystemMessagePromptTemplate.from_template(MP_SYSTEM_TEMPLATE)
    system_message = system_template.format()

    initial_data = {"Deidentified_Free_Text_Clinical_Notes": note, "Discharge_medications": discharge_medications, "Error_Type": error_type, "Error_Type_Explanation": error_type_explanation, "Error_Injection_Strategy": error_injection_strategy, "Failed_Cases_and_Reasons": failed_cases_and_reasons}

    state_MP = {"messages": [system_message], "tool_retry_count": 0, "initial_data": initial_data, "referenced_drug_information": "", "search_arg": ""}

    # Run the workflow
    config = {"configurable": {"thread_id": "1"}}
    final_state = graph.invoke(
        state_MP,
        config,
    )

    # Parse the output
    finally_out = final_state['messages'][-1]
    parser = JsonOutputParser(pydantic_object=Modification_outformat)
    try:
        json_object = parser.parse(finally_out.content)
    except Exception as e:
        print(e)

    search_arg = final_state["search_arg"]
    referenced_drug_information = final_state["referenced_drug_information"]

    return search_arg, referenced_drug_information, json_object["modified_discharge_medications"], json_object["modification_rationale"]

def Validation_phase(
        V_llm_model, 
        note: str, 
        discharge_medications: str, 
        error_type: str, 
        error_type_explanation: str, 
        referenced_drug_information: str, modified_discharge_medications: str, 
        modification_rationale: str
) -> tuple[str, str]:
    """
    Validation Phase: Check whether the modified prescription meets the specified error type requirements.

    Args:
        V_llm: The LLM model used in the validation phase.
        note: Clinical note.
        discharge_medications: Original discharge medications.
        error_type: Type of error.
        error_type_explanation: Explanation of the error type.
        referenced_drug_information: Reference drug information.
        modified_discharge_medications: Modified discharge medications.
        modification_rationale: Rationale for the modification.

    Returns:
        Tuple: (verification_result, failure_reason)
    """

    parser = JsonOutputParser(pydantic_object=Validation_outformat)
    system_template = SystemMessagePromptTemplate.from_template(VLP_SYSTEM_TEMPLATE)
    human_template = HumanMessagePromptTemplate.from_template(VLP_USER_TEMPLATE, partial_variables={"format_instructions": parser.get_format_instructions()})
    prompt_template = ChatPromptTemplate.from_messages([system_template, human_template])

    chain = prompt_template | V_llm_model | parser
    response = chain.invoke({"Deidentified_Free_Text_Clinical_Notes": note, "Error_Type": error_type, "Error_Type_Explanation": error_type_explanation, "Referenced_Drug_Information": referenced_drug_information, "Modified_Discharge_Medications": modified_discharge_medications, "Modification_Rationale": modification_rationale})
    return response['verification_result'], response['failure_reason'] 
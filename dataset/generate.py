import pandas as pd
import json
import os
import argparse
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from langchain_core.prompts import HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from utils import Modification_phase, Validation_phase, build_graph
from utils import MedicationError
import sys

def setup_environment(openfda_api_key):
    """Set up environment variables and paths"""
    # Add project root to path
    sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
    
    global openfda_tool
    from tool import openfda_tool
    
    # Set OpenFDA API key - this should be set in environment or passed as arg
    os.environ["OPENFDA_API_KEY"] = "openfda_api_key"

def load_checkpoint(checkpoint_file):
    """Load checkpoint the last processed."""   
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            data = json.load(f)
            return data.get("processed_errors", {}), data.get("last_row", 0)
    return {}, 0

def save_checkpoint(processed_errors, last_row, checkpoint_file):
    """Save checkpoint to file.""" 
    with open(checkpoint_file, "w") as f:
        json.dump({
            "processed_errors": processed_errors,
            "last_row": last_row
        }, f)

def save_output_file(error_type, data_list, output_dir):
    """Save collected error data to CSV file"""
    os.makedirs(output_dir, exist_ok=True)
    outputa_file = os.path.join(output_dir, f"{error_type.replace(' ', '_')}.csv") 
    df = pd.DataFrame(data_list)
    df.to_csv(outputa_file, index=False)

def extract_medical_content(text, model_name, base_url):
    """
    Extract clinical_note and discharge_medications from medical records using LLM.
    Args:
        text (str): Original medical record text.
        model_name (str): Model name for LLM.
        base_url (str): Base URL for LLM service.

    Returns:
        tuple: (clinical_note, discharge_med) - extracted clinical note and discharge medications.
    """

    class Extraction_outformat(BaseModel):
        clinical_note: str = Field(
        description="All information except for discharge medications"
    )
        discharge_medications: str = Field(
        description="List of discharge medications, extracted from the 'Discharge Medications:' part."
    )

    llm = ChatOpenAI(model=model_name,
                     base_url=base_url,
                     api_key="fake-key",
                     temperature=1
    )

    user_template = """
    You are a medical text analysis expert tasked with extracting two parts from the following medical record:

    1. **clinical_note**: Contains patient clinical information, including:
        - Patient basic information (e.g., "Name", "Sex")
        - Clinical descriptions (e.g., "Allergies:", "Allergies:"、"Chief Complaint:"、"Major Surgical or Invasive Procedure:"、"History of Present Illness:"、"Past Medical History:"、"Social History:"、"Brief Hospital Course:")
        - Diagnosis（e.g., "Discharge Diagnosis:"）
        - Other clinical information
    
        Exclude these sections:
        - Laboratory results (e.g., "Physical Exam:", "ADMISSION PHYSICAL EXAM:")
        - Imaging reports (e.g., "Imaging showed:")
        - Microbiology data (e.g., "MICROBIOLOGY:")
        - Physical examination details (e.g., "Physical Exam:")
        - Discharge instructions (e.g., "Discharge Instructions:", "Discharge Condition:")

    2. **discharge_medications**: Only extract the "Discharge Medications:" section content without additional text.

    Original Text:
    {text}


    {format_instructions}
    """
    parser = JsonOutputParser(pydantic_object=Extraction_outformat)
    
    user_template = HumanMessagePromptTemplate.from_template(
        user_template, 
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    prompt_template = ChatPromptTemplate.from_messages([user_template])

    chain = prompt_template | llm | parser
    response = chain.invoke({"text": text})
    print(f"Data extraction completed!")

    return response["clinical_note"], response["discharge_medications"] 

def generate_error(clinical_note, discharge_med, error_type, Modification_model_name, Validation_model_name, modification_base_url, validation_base_url):
    """
    Generate specified error type in discharge medications.

    Args:
        clinical_note (str): Clinical notes text
        discharge_med (str): Discharge prescription text
        error_type (str): Type of error to generate
        Modification_model_name (str): Model name for modification phase
        Validation_model_name (str): Model name for validation phase
        modification_base_url (str): Base URL for modification model service
        validation_base_url (str): Base URL for validation model service

    Returns:
        dict: Results containing error generation details, including the following keys:
            - sucess (bool): Whether the error generation was successful.
            - modified_discharge_medication (str): Modified discharge prescription.
            - modification_rationale (str): Reason for modification.
            - referenced_drug_information (str): Referenced drug information.
    """
    # The max number of records to be generated for each error type
    MAX_TRIES = 3
    verification_result = "false"


    # Setup models
    M_llm = ChatOpenAI(model=Modification_model_name, base_url=modification_base_url, api_key="fake-key", temperature=0.7)
    Mllm_with_tools = M_llm.bind_tools([openfda_tool])

    V_llm = ChatOpenAI(model=Validation_model_name, base_url=validation_base_url, api_key="fake-key", temperature=0.7)

    # Get error type definitions
    all_medication_error = MedicationError()
    error_type_explanation = all_medication_error.get_error_explanation(error_type)
    error_injection_strategy = all_medication_error.get_injection_strategy(error_type)   
    failed_cases_and_reasons = 'None'
    generate_result = {}

    n = 0
    while (verification_result.lower() == "false") and n <  MAX_TRIES:
        n += 1
        # Modification phase
        try:
            search_arg, referenced_drug_information, modified_discharge_medications, modification_rationale = Modification_phase(Mllm_with_tools, clinical_note, discharge_med, error_type, error_type_explanation, error_injection_strategy, failed_cases_and_reasons)
        except Exception as e:
            # print(e)
            generate_result['is_success'] = False
            generate_result['modified_discharge_medications'] = f'Failed to generate in Modification pharse with error: {str(e)}'
            generate_result['modification_rationale'] = f'Failed to generate in Modification pharse with error: {str(e)}'
            generate_result['referenced_drug_information'] = ''
            return generate_result

        # Validation pharse
        verification_result, failure_reason = Validation_phase(V_llm, clinical_note, discharge_med, error_type, error_type_explanation, referenced_drug_information, modified_discharge_medications, modification_rationale)

        if verification_result.lower()   == "false":
            failed_cases_and_reasons = (
                f'**Failed Case {n}**: \n\n '
                f'Failed Case Modified Discharge Medications:\n{modified_discharge_medications}\n'
                f'Failure Reason:\n{failure_reason}\n'
                f'Failed Case Search Argument In Case:\n{search_arg}\n'
            )
            print(f"Validation failed, retrying!\nFailure reason: {failed_cases_and_reasons}，")
        else:
            print(f"Validation successful!")

    if verification_result.lower() == "false":
        generate_result['is_success'] = False
        generate_result['modified_discharge_medications'] = 'Failed to generate in Validation pharse.'
        generate_result['modification_rationale'] = 'Failed to generate in Validation pharse.'
    else:
        generate_result['is_success'] = True
        generate_result['modified_discharge_medications'] = modified_discharge_medications
        generate_result['modification_rationale'] = modification_rationale
        
    generate_result['referenced_drug_information'] = str(referenced_drug_information)
    
    return generate_result

def generate_medication_errors(input_csv, 
                               output_dir, 
                               checkpoint_file, 
                               max_records_per_file, 
                               Modification_model_name, 
                               Validation_model_name, 
                               openfda_api_key, 
                               modification_base_url, 
                               validation_base_url):
    """
    Main function to generate medication errors in medical records.
    
    Args:
        input_csv (str): Path to input CSV file
        output_dir (str): Directory for output files
        checkpoint_file (str): Path to checkpoint file 
        max_records_per_file (int): Maximum number of records per output file
        Modification_model_name (str): Model name for modification phase
        Validation_model_name (str): Model name for validation phase
        openfda_api_key (str): OpenFDA API key
        modification_base_url (str): Base URL for modification model service
        validation_base_url (str): Base URL for validation model service
    """    
    # Setup environment
    setup_environment(openfda_api_key)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load checkpoint
    processed_errors, start_row = load_checkpoint(checkpoint_file)

    # Read input data
    raw_data = pd.read_csv(input_csv)

        # Define error types
    MEDICATION_ERROR_TYPES = [
        'Inappropriate Indication', 
        'Failure to Consider Allergy', 
        'Failure to Consider Medical History', 
        'Inappropriate Dosage Form', 
        'Drug-Drug Interactions', 
        'Dosage Inconsistency', 
        'Incorrect Administration Route', 
    ]

    # Generate data for each error type
    for error_type in MEDICATION_ERROR_TYPES:
        # Skip if already reached target count
        current_count = processed_errors.get(error_type, 0)
        if current_count >= max_records_per_file:
            print(f"Skipping {error_type}: alread has  {current_count} records")
            continue

        # Load existing output data if file exists
        output_file = os.path.join(output_dir, f"{error_type.replace(' ', '_')}.csv")
        output_data = []
        if os.path.exists(output_file):
            output_data = pd.read_csv(output_file).to_dict("records")
        
        # Process records starting from last position
        for i in range(start_row, len(raw_data)):
            row = raw_data.iloc[i]
            note_id = row["note_id"]
            subject_id = row["subject_id"]
            hadm_id = row["hadm_id"] 
            text = row["text"]

            # Extract medical content
            clinical_note, discharge_med = extract_medical_content(
                text, Modification_model_name, modification_base_url
            )

            # Generate error data
            result = generate_error(
                clinical_note, discharge_med, error_type,
                Modification_model_name, Validation_model_name,
                modification_base_url, validation_base_url)

            # Record if the error generation was successful
            if result["is_success"]:
                output_entry = {
                    "note_id": note_id,
                    "subject_id": subject_id,
                    "hadm_id": hadm_id,
                    "clinical_note": clinical_note,
                    "discharge_medications": discharge_med,
                    "modified_prescription": result["modified_discharge_medications"],
                    "rationale": result["modification_rationale"],
                    "referenced_drug_information": result["referenced_drug_information"]
                }
                output_data.append(output_entry)

                # Update the processed errors count and checkpoint
                current_count += 1
                processed_errors[error_type] = current_count
                save_checkpoint(processed_errors, i + 1)

                print(f"Processed {error_type}: {current_count}/{max_records_per_file}")

                # Save results
                save_output_file(error_type, output_data)

                # Stop if reached target count
                if current_count >= max_records_per_file:
                    break
            else:
                continue

        # Save any collected data even if not full
        if output_data:
            save_output_file(error_type, output_data)

    # Check if all error types are complete
    if all(processed_errors.get(et, 0) >= max_records_per_file for et in MEDICATION_ERROR_TYPES):
        print("All error types completed!")
        os.remove(max_records_per_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process medical records to inject medication errors.')
    parser.add_argument('--input_csv', type=str, required=True,
                        help='Path to input CSV file containing medical records from MIMIC-IV-Note')
    parser.add_argument('--output_dir', type=str, default='output_errors/',
                        help='Directory to store output files')
    parser.add_argument('--checkpoint_file', type=str, default='checkpoint.json',
                        help='Path to checkpoint file, used to resume from last processed record')
    parser.add_argument('--max_records_per_file', type=int, default=100,
                        help='Maximum number of records per output file')
    parser.add_argument('--Modification_model_name', type=str, required=True,
                        help='Modification model name deployed locally by VLLM')
    parser.add_argument('--Validation_model_name', type=str, required=True,
                        help='Validation model name deployed locally by VLLM')
    parser.add_argument('--openfda_api_key', type=str, required=True,
                        help='OpenFDA API key for drug information retrieval')
    parser.add_argument('--modification_base_url', type=str, required=True,
                        help='Base URL for modification model service')
    parser.add_argument('--validation_base_url', type=str, required=True,
                        help='Base URL for validation model service')

    args = parser.parse_args()   


    while True:
        try:
            generate_medication_errors(
                args.input_csv,
                args.output_dir,
                args.checkpoint_file,
                args.max_records_per_file,
                args.Modification_model_name,
                args.Validation_model_name,
                args.openfda_api_key,
                args.modification_base_url,
                args.validation_base_url
            )
            break  # Exit loop if successful
        except Exception as e:
            print(f"Unexpected interruption: {e}. Retrying...")
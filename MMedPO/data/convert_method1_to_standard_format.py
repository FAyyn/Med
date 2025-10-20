#!/usr/bin/env python3
"""
Convert Method 1 (Visual Indirect) data to standard DPO format.

Method 1 data characteristics:
- preferred: answer with full image context
- dispreferred: answer with background-only (masked) image
- Both use same question but different visual inputs
"""

import json
import os
from typing import Dict, Any

def extract_relative_path(image_path):
    """Extract relative path from full image path (e.g., xmlab1/source.jpg)."""
    # Extract the relative path from the full path
    # Example: /workspace/MMedPO/datasets/SLAKE/imgs/xmlab1/source.jpg -> xmlab1/source.jpg
    if '/imgs/' in image_path:
        return image_path.split('/imgs/')[-1]
    else:
        # Fallback: extract last two parts of the path
        parts = image_path.split('/')
        if len(parts) >= 2:
            return '/'.join(parts[-2:])
        else:
            return os.path.basename(image_path)

def generate_rejected_answer_for_method1(pref_answer, disp_answer, question):
    """
    Generate a distinct rejected answer for Method 1 data.
    
    Method 1 compares answers with full image vs background-only image.
    When answers are identical, we create visually-limited uncertain responses.
    """
    if pref_answer.strip() == disp_answer.strip():
        # Create visually-limited rejected answers based on question type
        question_lower = question.lower()
        
        if any(keyword in question_lower for keyword in ['modality', 'imaging', 'scan', 'technique']):
            return "I cannot clearly determine the imaging modality from the background alone."
        elif any(keyword in question_lower for keyword in ['body part', 'part of', 'belong to', 'region', 'anatomy']):
            return "The anatomical region is not clearly identifiable without the main structures visible."
        elif any(keyword in question_lower for keyword in ['contain', 'see', 'visible', 'present']):
            if 'yes' in pref_answer.lower():
                return "I'm not certain, the structure might be present but is not clearly visible in this view."
            else:
                return "I cannot definitively determine if this structure is present without clearer visual details."
        elif any(keyword in question_lower for keyword in ['where', 'location', 'position']):
            return "I cannot accurately determine the location without the main anatomical structures visible."
        elif any(keyword in question_lower for keyword in ['healthy', 'normal', 'abnormal']):
            return "I cannot assess the health status without seeing the complete anatomical structures."
        elif any(keyword in question_lower for keyword in ['disease', 'condition', 'pathology']):
            return "I cannot identify specific diseases or conditions without clear visual details of the main structures."
        elif any(keyword in question_lower for keyword in ['main', 'largest', 'primary', 'organ']):
            return "I cannot clearly identify the main organ without the complete visual information."
        elif any(keyword in question_lower for keyword in ['bigger', 'larger', 'compare', 'size']):
            return "I cannot make accurate size comparisons without seeing the complete structures."
        elif any(keyword in question_lower for keyword in ['weighting', 'weighted', 'mr', 'mri']):
            return "I cannot determine the MR weighting without clear visual details of the main structures."
        else:
            return "I cannot provide a confident answer without seeing the complete anatomical structures."
    else:
        # If answers are different, use the disp answer as rejected
        return disp_answer

def convert_method1_to_standard_format(input_file: str, output_file: str):
    """Convert Method 1 data to standard DPO format."""
    
    converted_data = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                
                # Extract basic information
                question = data['question']
                pref_answer = data['pref']['answer']
                disp_answer = data['disp']['answer']
                # Use the full image path from pref (full image)
                image_path = data['pref']['image_path']
                weight = data['weight']
                
                # Generate rejected answer based on disp content
                rejected_answer = generate_rejected_answer_for_method1(pref_answer, disp_answer, question)
                
                # Round weight to 2 decimal places
                rounded_weight = round(weight, 2)
                
                # Create standard format entry
                converted_entry = {
                    "id": line_num - 1,  # Use numerical ID starting from 0
                    "image": extract_relative_path(image_path),
                    "conversations": [
                        {
                            "from": "human",
                            "value": f"{question}"  # Question already contains <image> token
                        },
                        {
                            "from": "gpt",
                            "value": pref_answer
                        }
                    ],
                    "rejected_conversations": [
                        {
                            "from": "human", 
                            "value": f"{question}"  # Question already contains <image> token
                        },
                        {
                            "from": "gpt",
                            "value": rejected_answer
                        }
                    ],
                    "weighted_score": rounded_weight
                }
                
                converted_data.append(converted_entry)
                
                # Print progress every 100 items
                if line_num % 100 == 0:
                    print(f"Processed {line_num} items...")
                    
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
            except KeyError as e:
                print(f"Missing key in line {line_num}: {e}")
                continue
    
    # Save converted data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nConversion completed!")
    print(f"Total items processed: {len(converted_data)}")
    print(f"Output saved to: {output_file}")
    
    # Show sample conversion
    if converted_data:
        sample = converted_data[0]
        print(f"\nSample conversion:")
        print(f"ID: {sample['id']}")
        print(f"Image: {sample['image']}")
        print(f"Question: {sample['conversations'][0]['value']}")
        print(f"Preferred Answer: {sample['conversations'][1]['value']}")
        print(f"Rejected Answer: {sample['rejected_conversations'][1]['value']}")
        print(f"Weighted Score: {sample['weighted_score']}")

if __name__ == "__main__":
    input_file = "/workspace/MMedPO/MMedPO/outputs/combined_vqa_rad/dpo_pairs_visual_indirect.jsonl"
    output_file = "/workspace/MMedPO/MMedPO/data/tie_dpo_dataset_method1_vqa_rad.json"
    
    convert_method1_to_standard_format(input_file, output_file)
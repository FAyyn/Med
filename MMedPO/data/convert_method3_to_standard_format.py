#!/usr/bin/env python3
"""
Convert Method 3 (Text Contrast) DPO data to standard format.

Method 3 compares the same answer under different text context conditions:
- pref: answer with text context (full_with_context)
- disp: answer without text context (full_without_context)

For DPO training, we need to create meaningful differences between preferred and rejected answers.
"""

import json
import os
from pathlib import Path

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

def generate_rejected_answer_for_method3(pref_answer, disp_answer, question):
    """
    Generate a distinct rejected answer for Method 3 data.
    
    Method 3 compares answers with vs without text context.
    When answers are identical, we create context-specific uncertain responses.
    """
    if pref_answer.strip() == disp_answer.strip():
        # Create context-specific rejected answers based on question type
        question_lower = question.lower()
        
        if any(keyword in question_lower for keyword in ['modality', 'imaging', 'scan', 'technique']):
            return "I cannot clearly determine the imaging modality without additional context."
        elif any(keyword in question_lower for keyword in ['body part', 'part of', 'belong to', 'region', 'anatomy']):
            return "The anatomical region is not clearly identifiable without textual context."
        elif any(keyword in question_lower for keyword in ['contain', 'see', 'visible', 'present']):
            if 'yes' in pref_answer.lower():
                return "I'm not certain, the structure might be present but is difficult to confirm without context."
            else:
                return "I cannot definitively determine if this structure is present without additional information."
        elif any(keyword in question_lower for keyword in ['where', 'location', 'position']):
            return "I cannot accurately determine the location without textual context clues."
        elif any(keyword in question_lower for keyword in ['healthy', 'normal', 'abnormal']):
            return "I cannot assess the health status without proper contextual information."
        elif any(keyword in question_lower for keyword in ['disease', 'condition', 'pathology']):
            return "I cannot identify specific diseases or conditions without textual context."
        elif any(keyword in question_lower for keyword in ['main', 'largest', 'primary', 'organ']):
            return "I cannot clearly identify the main organ without additional context."
        elif any(keyword in question_lower for keyword in ['bigger', 'larger', 'compare', 'size']):
            return "I cannot make accurate size comparisons without contextual information."
        elif any(keyword in question_lower for keyword in ['weighting', 'weighted', 'mr', 'mri']):
            return "I cannot determine the MR weighting without textual context."
        else:
            return "I cannot provide a confident answer without additional textual context."
    else:
        # If answers are different, use the disp answer as rejected
        return disp_answer

def convert_method3_to_standard_format(input_file, output_file):
    """Convert Method 3 JSONL data to standard DPO JSON format."""
    
    converted_data = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                item = json.loads(line.strip())
                
                # Extract basic information
                question = item.get('question_with_context', item.get('question_without_context', ''))
                pref_answer = item['pref']['answer']
                disp_answer = item['disp']['answer']
                image_path = item['images']['image_path']
                weight = round(item['weight'], 2)  # Round to 2 decimal places
                
                # Generate rejected answer
                rejected_answer = generate_rejected_answer_for_method3(pref_answer, disp_answer, question)
                
                # Create standard format entry
                converted_entry = {
                    "id": line_num - 1,  # Use numerical ID starting from 0
                    "image": extract_relative_path(image_path),
                    "conversations": [
                        {
                            "from": "human",
                            "value": f"<image>\n{question}"
                        },
                        {
                            "from": "gpt",
                            "value": pref_answer
                        }
                    ],
                    "rejected_conversations": [
                        {
                            "from": "human", 
                            "value": f"<image>\n{question}"
                        },
                        {
                            "from": "gpt",
                            "value": rejected_answer
                        }
                    ],
                    "weighted_score": weight
                }
                
                converted_data.append(converted_entry)
                
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
            except KeyError as e:
                print(f"Missing key in line {line_num}: {e}")
                continue
    
    # Save converted data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, indent=2, ensure_ascii=False)
    
    print(f"Conversion completed!")
    print(f"Processed {len(converted_data)} items from {input_file}")
    print(f"Saved converted data to {output_file}")
    
    # Show sample
    if converted_data:
        print(f"\nSample converted item:")
        sample = converted_data[0]
        print(f"ID: {sample['id']}")
        print(f"Image: {sample['image']}")
        print(f"Question: {sample['conversations'][0]['value']}")
        print(f"Preferred Answer: {sample['conversations'][1]['value']}")
        print(f"Rejected Answer: {sample['rejected_conversations'][1]['value']}")
        print(f"Weighted Score: {sample['weighted_score']}")

if __name__ == "__main__":
    input_file = "/workspace/MMedPO/MMedPO/outputs/combined_round2/dpo_pairs_text_contrast.jsonl"
    output_file = "/workspace/MMedPO/MMedPO/data/tie_dpo_dataset_method3_round3.json"
    
    convert_method3_to_standard_format(input_file, output_file)
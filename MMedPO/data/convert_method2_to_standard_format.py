#!/usr/bin/env python3
"""
Convert Method 2 (Visual Consistency) data to standard DPO format.

Method 2 data characteristics:
- preferred: accurate, complete answers based on full image
- dispreferred: incomplete, incorrect, or irrelevant answer fragments
- Both use same visual input but differ in answer quality
"""

import json
import random
import os
from typing import Dict, Any

def generate_rejected_answer(question: str, dispreferred_answer: str) -> str:
    """
    Generate rejected answer based on question type and dispreferred content.
    For Method 2, we enhance the dispreferred answer to make it more coherent
    while maintaining its lower quality.
    """
    question_lower = question.lower()
    
    # If dispreferred answer is very short or incomplete, enhance it
    if len(dispreferred_answer.strip()) < 10 or dispreferred_answer.strip().endswith('.'):
        if any(keyword in question_lower for keyword in ['abnormal', 'disease', 'healthy']):
            return "I'm not entirely certain about the health status based on this image."
        elif any(keyword in question_lower for keyword in ['what', 'which', 'where']):
            return "The specific details are not clearly visible in this image."
        elif any(keyword in question_lower for keyword in ['how', 'treat', 'prevent']):
            return "Treatment options would require more detailed clinical assessment."
        else:
            return "I cannot provide a definitive answer based on this image alone."
    
    # For longer dispreferred answers, use them as is but ensure they're complete sentences
    enhanced_answer = dispreferred_answer.strip()
    if not enhanced_answer.endswith('.'):
        enhanced_answer += "."
    
    # Add uncertainty markers for health-related questions
    if any(keyword in question_lower for keyword in ['disease', 'abnormal', 'healthy', 'cancer']):
        if not any(marker in enhanced_answer.lower() for marker in ['uncertain', 'unclear', 'difficult', 'cannot']):
            enhanced_answer = "Based on the image, " + enhanced_answer
    
    return enhanced_answer

def preserve_image_path(image_path):
    """Preserve the complete image path."""
    return image_path

def convert_method2_to_standard_format(input_file: str, output_file: str):
    """Convert Method 2 data to standard DPO format."""
    
    converted_data = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                
                # Extract basic information
                qid = data['qid']
                image = data['image']
                question = data['question']
                preferred_answer = data['preferred']['answer']
                dispreferred_answer = data['dispreferred']['answer']
                weight = data['weight']
                
                # Generate rejected answer based on dispreferred content
                rejected_answer = generate_rejected_answer(question, dispreferred_answer)
                
                # Round weight to 2 decimal places
                rounded_weight = round(weight, 2)
                
                # Create standard format entry
                converted_entry = {
                    "id": line_num - 1,  # Use numerical ID starting from 0
                    "image": preserve_image_path(image),
                    "conversations": [
                        {
                            "from": "human",
                            "value": f"<image>\n{question}"
                        },
                        {
                            "from": "gpt",
                            "value": preferred_answer
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
        print(f"Preferred Answer: {sample['conversations'][1]['value']}")
        print(f"Rejected Answer: {sample['rejected_conversations'][1]['value']}")
        print(f"Weighted Score: {sample['weighted_score']}")

if __name__ == "__main__":
    input_file = "/workspace/MMedPO/MMedPO/outputs/combined_vqa_rad/dpo_pairs_visual_consistency.jsonl"
    output_file = "/workspace/MMedPO/MMedPO/data/tie_dpo_dataset_method2_vqa_rad.json"
    
    convert_method2_to_standard_format(input_file, output_file)
#!/usr/bin/env python3
"""
Aligned converter for Method 2 (Visual Consistency) to standard DPO format.

Adjustments vs original:
- Ensure question formatting uses single <image> prefix only.
- Preserve image path exactly as provided (these are filenames in Method 2).
- Keep original script unmodified; this is a new aligned version.
"""

import json
from typing import Dict, Any


def normalize_question(question: str) -> str:
    """Ensure the question has a single '<image>' token at the beginning."""
    q = question.strip()
    # Remove potential duplicate leading <image> tokens
    if q.lower().startswith("<image>"):
        q = q[len("<image>"):].lstrip()
    # Reconstruct with a single <image> prefix and newline
    return f"<image>\n{q}"


def preserve_image_path(image_path: str) -> str:
    """Preserve complete image path or filename as provided."""
    return image_path


def convert_method2_to_standard_format_aligned(input_file: str, output_file: str):
    """Convert Method 2 data to aligned standard DPO format."""
    converted_data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                qid = data['qid']
                image = data['image']
                question = data['question']
                preferred_answer = data['preferred']['answer']
                dispreferred_answer = data['dispreferred']['answer']
                weight = data['weight']

                # Skip items where preferred and dispreferred answers are identical
                if preferred_answer.strip() == dispreferred_answer.strip():
                    continue

                # Normalize question to single <image> prefix
                question_norm = normalize_question(question)
                rejected_answer = dispreferred_answer
                rounded_weight = round(weight, 2)

                converted_entry = {
                    "id": line_num - 1,
                    "image": image,
                    "conversations": [
                        {"from": "human", "value": question_norm},
                        {"from": "gpt", "value": preferred_answer}
                    ],
                    "rejected_conversations": [
                        {"from": "human", "value": question_norm},
                        {"from": "gpt", "value": rejected_answer}
                    ],
                    "weighted_score": rounded_weight
                }
                converted_data.append(converted_entry)
                if line_num % 100 == 0:
                    print(f"Processed {line_num} items...")
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
            except KeyError as e:
                print(f"Missing key in line {line_num}: {e}")
                continue

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)

    print(f"\nConversion completed!")
    print(f"Total items processed: {len(converted_data)}")
    print(f"Output saved to: {output_file}")

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
    output_file = "/workspace/MMedPO/MMedPO/data/tie_dpo_dataset_method2_vqa_rad_aligned.json"
    convert_method2_to_standard_format_aligned(input_file, output_file)
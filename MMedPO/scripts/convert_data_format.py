#!/usr/bin/env python3
"""Convert data format from conversations to question-answer format for inference."""

import json
import argparse

def convert_conversations_to_qa(input_file, output_file):
    """Convert conversations format to question-answer format."""
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    qa_data = []
    for item in data:
        # Extract the first human question and the corresponding answer
        conversations = item['conversations']
        
        # Find the first human question
        human_question = None
        gpt_answer = None
        
        for i, conv in enumerate(conversations):
            if conv['from'] == 'human':
                human_question = conv['value']
                # Look for the next GPT response
                if i + 1 < len(conversations) and conversations[i + 1]['from'] == 'gpt':
                    gpt_answer = conversations[i + 1]['value']
                break
        
        if human_question and gpt_answer:
            # Remove <image> token from the question
            question_text = human_question.replace('<image>\n', '').replace('<image>', '').strip()
            
            qa_item = {
                "question_id": item['id'],
                "image": item['image'],
                "text": question_text,
                "reference_answer": gpt_answer.strip()
            }
            qa_data.append(qa_item)
    
    # Write as JSONL
    with open(output_file, 'w') as f:
        for item in qa_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Converted {len(qa_data)} questions from {input_file} to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert data format for inference")
    parser.add_argument("--input", type=str, default="/workspace/MMedPO/data/slake_dpo_weighted.json")
    parser.add_argument("--output", type=str, default="/workspace/MMedPO/data/slake_questions.jsonl")
    args = parser.parse_args()
    
    convert_conversations_to_qa(args.input, args.output)
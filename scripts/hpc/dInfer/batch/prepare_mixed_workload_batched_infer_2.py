import argparse
import json
import os
import random
from datasets import load_dataset
from jinja2 import Template
from transformers import AutoTokenizer

TASKS = {
    "mmlu": {"path": "cais/mmlu", "name": "all", "split": "test", "gen_len": 32, "type": "MC", "count": 200},
    "piqa": {"path": "piqa", "name": None, "split": "validation", "gen_len": 32, "type": "MC", "count": 200},
    "hellaswag": {"path": "hellaswag", "name": "default", "split": "validation", "gen_len": 32, "type": "MC", "count": 200},
    "arc_c": {"path": "allenai/ai2_arc", "name": "ARC-Challenge", "split": "test", "gen_len": 32, "type": "MC", "count": 200},
    "gsm8k": {"path": "gsm8k", "name": "main", "split": "test", "gen_len": 256, "type": "GEN", "count": 50},
    "ifeval": {
        "path": "parquet",
        "name": None,
        "split": "train",
        "data_files": "https://huggingface.co/datasets/google/IFEval/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet",
        "gen_len": 256,
        "type": "GEN",
        "count": 50
    }
}

GSM8K_FEWSHOT = (
    "Question: Angelo and Melanie want to plan how many hours over the next week they should study together for their test next week. They have 2 chapters of their textbook to study and 4 worksheets to memorize. They figure out that they should dedicate 3 hours to each chapter of their textbook and 1.5 hours for each worksheet. If they plan to study no more than 4 hours each day, how many days should they plan to study total over the next week if they take a 10-minute break every hour, include 3 10-minute snack breaks each day, and 30 minutes for lunch each day?\n"
    "Let's think step by step\n"
    "Answer: Angelo and Melanie think they should dedicate 3 hours to each of the 2 chapters, 3 hours x 2 chapters = 6 hours total.\n"
    "For the worksheets they plan to dedicate 1.5 hours for each worksheet, 1.5 hours x 4 worksheets = 6 hours total.\n"
    "Angelo and Melanie need to start with planning 12 hours to study, at 4 hours a day, 12 / 4 = 3 days.\n"
    "However, they need to include time for breaks and lunch. Every hour they want to include a 10-minute break, so 12 total hours x 10 minutes = 120 extra minutes for breaks.\n"
    "They also want to include 3 10-minute snack breaks, 3 x 10 minutes = 30 minutes.\n"
    "And they want to include 30 minutes for lunch each day, so 120 minutes for breaks + 30 minutes for snack breaks + 30 minutes for lunch = 180 minutes, or 180 / 60 minutes per hour = 3 extra hours.\n"
    "So Angelo and Melanie want to plan 12 hours to study + 3 hours of breaks = 15 hours total.\n"
    "They want to study no more than 4 hours each day, 15 hours / 4 hours each day = 3.75\n"
    "They will need to plan to study 4 days to allow for all the time they need.\n"
    "The answer is 4\n\n"
    "Question: Mark's basketball team scores 25 2 pointers, 8 3 pointers and 10 free throws.  Their opponents score double the 2 pointers but half the 3 pointers and free throws.  What's the total number of points scored by both teams added together?\n"
    "Let's think step by step\n"
    "Answer: Mark's team scores 25 2 pointers, meaning they scored 25*2= 50 points in 2 pointers.\n"
    "His team also scores 6 3 pointers, meaning they scored 8*3= 24 points in 3 pointers\n"
    "They scored 10 free throws, and free throws count as one point so they scored 10*1=10 points in free throws.\n"
    "All together his team scored 50+24+10= 84 points\n"
    "Mark's opponents scored double his team's number of 2 pointers, meaning they scored 50*2=100 points in 2 pointers.\n"
    "His opponents scored half his team's number of 3 pointers, meaning they scored 24/2= 12 points in 3 pointers.\n"
    "They also scored half Mark's team's points in free throws, meaning they scored 10/2=5 points in free throws.\n"
    "All together Mark's opponents scored 100+12+5=117 points\n"
    "The total score for the game is both team's scores added together, so it is 84+117=201 points\n"
    "The answer is 201\n\n"
    "Question: Bella has two times as many marbles as frisbees. She also has 20 more frisbees than deck cards. If she buys 2/5 times more of each item, what would be the total number of the items she will have if she currently has 60 marbles?\n"
    "Let's think step by step\n"
    "Answer: When Bella buys 2/5 times more marbles, she'll have increased the number of marbles by 2/560 = 24\n"
    "The total number of marbles she'll have is 60+24 = 84\n"
    "If Bella currently has 60 marbles, and she has two times as many marbles as frisbees, she has 60/2 = 30 frisbees.\n"
    "If Bella buys 2/5 times more frisbees, she'll have 2/530 = 12 more frisbees.\n"
    "The total number of frisbees she'll have will increase to 30+12 = 42\n"
    "Bella also has 20 more frisbees than deck cards, meaning she has 30-20 = 10 deck cards\n"
    "If she buys 2/5 times more deck cards, she'll have 2/5*10 = 4 more deck cards.\n"
    "The total number of deck cards she'll have is 10+4 = 14\n"
    "Together, Bella will have a total of 14+42+84 = 140 items\n"
    "The answer is 140\n\n"
    "Question: A group of 4 fruit baskets contains 9 apples, 15 oranges, and 14 bananas in the first three baskets and 2 less of each fruit in the fourth basket. How many fruits are there?\n"
    "Let's think step by step\n"
    "Answer: For the first three baskets, the number of apples and oranges in one basket is 9+15=24\n"
    "In total, together with bananas, the number of fruits in one basket is 24+14=38 for the first three baskets.\n"
    "Since there are three baskets each having 38 fruits, there are 3*38=114 fruits in the first three baskets.\n"
    "The number of apples in the fourth basket is 9-2=7\n"
    "There are also 15-2=13 oranges in the fourth basket\n"
    "The combined number of oranges and apples in the fourth basket is 13+7=20\n"
    "The fourth basket also contains 14-2=12 bananas.\n"
    "In total, the fourth basket has 20+12=32 fruits.\n"
    "The four baskets together have 32+114=146 fruits.\n"
    "The answer is 146\n\n"
)

PROMPT_TEMPLATES = {
    "mmlu": Template("Question: {{question}}\nOptions:\n{% for i in range(choices|length) %}{{ ['A', 'B', 'C', 'D'][i] }}: {{choices[i]}}\n{% endfor %}Answer: "),
    "piqa": Template("Question: {{goal}}\nOptions:\nA: {{sol1}}\nB: {{sol2}}\nAnswer: "),
    "hellaswag": Template("Question: Which is the most logical continuation of the context?\nContext: {{ctx}}\nOptions:\n0: {{endings[0]}}\n1: {{endings[1]}}\n2: {{endings[2]}}\n3: {{endings[3]}}\nAnswer: "),
    "arc_c": Template("Question: {{question}}\nOptions:\n{% for i in range(choices.label|length) %}{{choices.label[i]}}: {{choices.text[i]}}\n{% endfor %}Answer: "),
    "gsm8k": Template(GSM8K_FEWSHOT + "Question: {{question}}\nLet's think step by step\nAnswer:"),
    "ifeval": Template("{{prompt}}")
}

TARGET_TEMPLATES = {
    "mmlu": Template("{{ answer if answer is string else ['A', 'B', 'C', 'D'][answer] }}"),
    "piqa": Template("{{ answer if answer is defined else ('A' if label == 0 else 'B') }}"),
    "hellaswag": Template("{{ label }}"),
    "arc_c": Template("{{ answerKey }}"),
    "gsm8k": Template("{{ answer }}"),
    "ifeval": Template("")
}

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare mixed workload dataset for continuous batching evaluation.")
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="inclusionX/Ling-mini-2.0",
        help="Path or HuggingFace ID of the tokenizer to apply chat templates (e.g., for IFEval)."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/mixed_workload.jsonl",
        help="Path where the generated mixed workload JSONL will be saved."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for dataset sampling and shuffling."
    )
    return parser.parse_args()

def apply_chat_template(tokenizer, user_prompt):
    messages = [{"role": "user", "content": user_prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def main():
    args = parse_args()
    
    print(f"Loading tokenizer from: {args.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    
    mixed_dataset = []
    print(f"Building eval-ready mixed workload dataset with 4-shot GSM8K (seed={args.seed})...")
    
    for task_name, config in TASKS.items():
        print(f"Processing {task_name}...")
        
        if "data_files" in config:
            ds = load_dataset(config["path"], name=config["name"], split=config["split"], data_files=config["data_files"])
        else:
            ds = load_dataset(config["path"], name=config["name"], split=config["split"])
        
        ds = ds.shuffle(seed=args.seed)
        subset = ds.select(range(min(config["count"], len(ds))))
        
        prompt_template = PROMPT_TEMPLATES[task_name]
        target_template = TARGET_TEMPLATES[task_name]
        
        for row in subset:
            raw_prompt = prompt_template.render(**row)
            ground_truth = target_template.render(**row)
            
            if task_name == "ifeval":
                formatted_prompt = apply_chat_template(tokenizer, raw_prompt)
            else:
                formatted_prompt = raw_prompt

            mixed_dataset.append({
                "task": task_name,
                "task_type": config["type"],
                "prompt": formatted_prompt,
                "ground_truth": ground_truth,
                "gen_length": config["gen_len"],
                "raw_data": row
            })

    random.seed(args.seed)
    random.shuffle(mixed_dataset)
    
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    with open(args.output_path, "w", encoding="utf-8") as f:
        for item in mixed_dataset:
            f.write(json.dumps(item) + "\n")
            
    print(f"\nSaved {len(mixed_dataset)} mixed prompts to {args.output_path}")

if __name__ == "__main__":
    main()
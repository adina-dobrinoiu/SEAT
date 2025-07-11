import argparse
import json
import random
import yaml
from tqdm import tqdm
from pathlib import Path
from components.data_processing import load_data
from components.knn import KNN
from components.model import LlamaModelForSequenceCompletion
from components.prompt import get_prompt_per_task, get_few_shot_examples_per_task, get_few_shot_examples_all_tasks, \
    get_prompt_all_tasks

TASKS = ["argument", "emotion", "sentiment", "topic", "values"]
ANNOTATORS = ["Annotator_1", "Annotator_2", "Annotator_3", "Annotator_4", "Annotator_5"]
CONFIG_PATH = Path("./config.yaml")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, required=True, help="Number of few-shot examples (K)")
    parser.add_argument("--mode", choices=["all", "per", "baseline"], required=True, help="Prompting mode: all tasks or per task")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()

def load_config(path: Path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    args = parse_args()
    config = load_config(CONFIG_PATH)

    random.seed(args.seed)

    # Load data
    data = load_data()

    # Load KNN for few-shot examples
    knn_classifier = KNN(k=args.k)
    train_dataset = [entry['sentence'] for entry in data]
    knn_classifier.fit(train_dataset)

    # Load model and initialize pipeline
    llama_model = LlamaModelForSequenceCompletion(config)
    llama_model.init_pipeline()
    pipe = llama_model.pipeline

    if args.k == 0:
        for annotator in ANNOTATORS:
            results = []

            for obj in tqdm(data):
                sentence = obj['sentence']

                prompt = get_prompt_per_task(
                    sentence=sentence,
                    task="",
                    previous_task_annotations="",
                    few_shot_examples="",
                    k=args.k
                )

                generated = pipe(prompt.strip())[0]['generated_text'][len(prompt.strip()):]

                results.append({
                    "sentence": sentence,
                    "generated_response": generated,
                    "examples": ""
                })

            save_path = Path(f"./results/results_{annotator}_k{args.k}_seed{args.seed}.json")
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4)
            print(f"Saved: {save_path}")
    else:
        if args.mode == "per":
            for task in TASKS[:4]:  # exclude 'values'
                for annotator in ANNOTATORS:
                    results = []

                    for obj in tqdm(data):
                        sentence = obj['sentence']
                        previous_annotation = obj[task][annotator]

                        few_shot = get_few_shot_examples_per_task(
                            sentence, annotator, task, knn_classifier, data
                        )

                        prompt = get_prompt_per_task(
                            sentence=sentence,
                            task=task,
                            previous_task_annotations=previous_annotation,
                            few_shot_examples=few_shot,
                            k=args.k
                        )

                        generated = pipe(prompt.strip())[0]['generated_text'][len(prompt.strip()):]

                        results.append({
                            "sentence": sentence,
                            "generated_response": generated,
                            "examples": few_shot
                        })

                    save_path = Path(f"./results/results_{task}_{annotator}_k{args.k}_{args.mode}_seed{args.seed}.json")
                    with open(save_path, "w", encoding="utf-8") as f:
                        json.dump(results, f, indent=4)
                    print(f"Saved: {save_path}")

        elif args.mode == "all":
            for annotator in ANNOTATORS:
                results = []

                for obj in tqdm(data):
                    sentence = obj['sentence']
                    prev_annotations = [{t: obj[t][annotator]} for t in TASKS[:4]]  # exclude 'values'

                    few_shot = get_few_shot_examples_all_tasks(
                        sentence, annotator, knn_classifier, data
                    )

                    prompt = get_prompt_all_tasks(
                        sentence=sentence,
                        tasks=TASKS[:4],
                        previous_task_annotations=prev_annotations,
                        few_shot_examples=few_shot,
                        k=args.k
                    )

                    generated = pipe(prompt.strip())[0]['generated_text'][len(prompt.strip()):]

                    results.append({
                        "sentence": sentence,
                        "generated_response": generated,
                        "examples": few_shot
                    })

                save_path = Path(f"./results/results_all_{annotator}_k{args.k}_{args.mode}_seed{args.seed}.json")
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=4)
                print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()

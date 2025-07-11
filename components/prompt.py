VALUES_LEVEL_1 = ['Be creative', 'Be curious', 'Have freedom of thought', 'Be choosing own goals', 'Be independent',
          'Have freedom of action', 'Have privacy', 'Have an exciting life', 'Have a varied life', 'Be daring',
          'Have pleasure', 'Be ambitious', 'Have success', 'Be capable', 'Be intellectual', 'Be courageous',
          'Have influence', 'Have the right to command', 'Have wealth', 'Have social recognition',
          'Have a good reputation', 'Have a sense of belonging', 'Have good health', 'Have no debts',
          'Be neat and tidy', 'Have a comfortable life', 'Have a safe country', 'Have a stable society',
          'Be respecting traditions', 'Be holding religious faith', 'Be compliant', 'Be self-disciplined',
          'Be behaving properly', 'Be polite', 'Be honoring elders', 'Be humble', 'Have life accepted as is',
          'Be helpful', 'Be honest', 'Be forgiving', 'Have the own family secured', 'Be loving', 'Be responsible',
          'Have loyalty towards friends', 'Have equality', 'Be just', 'Have a world at peace',
          'Be protecting the environment', 'Have harmony with nature', 'Have a world of beauty', 'Be broadminded',
          'Have the wisdom to accept others', 'Be logical', 'Have an objective view']
VALUES = [
    "Self-direction: thought",
    "Self-direction: action",
    "Stimulation",
    "Hedonism",
    "Achievement",
    "Power: dominance",
    "Power: resources",
    "Face",
    "Security: personal",
    "Security: societal",
    "Tradition",
    "Conformity: rules",
    "Conformity: interpersonal",
    "Humility",
    "Benevolence: caring",
    "Benevolence: dependability",
    "Universalism: concern",
    "Universalism: nature",
    "Universalism: tolerance",
    "Universalism: objectivity"
]
TASKS = ["argument", "emotion", "sentiment", "topic", "values"]


def get_prompt_per_task(sentence, task, previous_task_annotations, few_shot_examples, k):
    # Introduction with conditional task reference
    intro = (
        f"You are an expert value annotator. Your task is to extract the most relevant value labels from a given sentence. The output should contain only the list of relevant values for the given sentence, without any extra information."
    )
    if k != 0:
        intro += f" You should also consider your previous annotations on the {task} detection task."

    # Sentence block
    sentence_block = f"\n\nSentence:\n{sentence}\n"

    # Previous annotation section (include only if k != 0)
    prev_ann_section = (
        f"\n{task.capitalize()} Annotations for this sentence:\n{previous_task_annotations}\n"
        if k != 0 else ""
    )

    # Few-shot section (include only if k > 1)
    few_shot_section = (
        f"\nFew Shot Examples:\n{few_shot_examples}\n"
        if k > 1 else ""
    )

    # Instructions (same for all cases)
    instructions = f"""
    Predefined Values:
    {VALUES}

    Instructions:
    - Return only the list of value labels relevant to the sentence.
    - Do not include any values not present in the list.
    - Format the result as a list of strings in JSON-like format, e.g.:
      ["Self-direction: thought", "Self-direction: action", "Stimulation"]
    - Do not include any code, explanations, formatting examples, headings, or any surrounding text.
    - Do not preface your answer with phrases like "Here is", "The values are", or anything similar.
    - Do not describe or explain your reasoning.

     Most important rule: Your entire response must consist of only the list of relevant value labels without any information.
"""

    return intro + sentence_block + prev_ann_section + few_shot_section + instructions


def get_prompt_all_tasks(sentence, tasks, previous_task_annotations, few_shot_examples, k):
    # Conditionally include task references in the intro
    formatted_tasks = ", ".join(tasks[:-2]) + " and " + tasks[-2] if len(tasks) >= 3 else ", ".join(tasks)

    intro = (
        "You are an expert value annotator. Your task is to extract the most relevant value labels from a given sentence. The output should contain only the list of relevant values for the given sentence, without any extra information."
    )
    if k != 0:
        intro += (
            f" You should also consider your previous annotations on the {formatted_tasks} detection tasks."
        )

    # Sentence block
    sentence_block = f"\n\nSentence:\n{sentence}\n"

    # Previous annotation block (shown only if k != 0)
    if k != 0:
        prev_lines = "\n".join(
            f"- {list(d.keys())[0].capitalize()}: {list(d.values())[0]}"
            for d in previous_task_annotations
        )
        annotation_block = (
            f"\nAnnotations for this sentence:\n{prev_lines}\n"
        )
    else:
        annotation_block = ""

    # Few-shot examples block (only if k > 1)
    few_shot_block = f"\nFew Shot Examples:\n{few_shot_examples}\n" if k > 1 else ""

    # Final instructions
    instructions = f"""
    Predefined Values:
    {VALUES}

    Instructions:
    - Return only the list of value labels relevant to the sentence.
    - Do not include any values not present in the list.
    - Format the result as a list of strings in JSON-like format, e.g.:
      ["Self-direction: thought", "Self-direction: action", "Stimulation"]
    - Do not include any code, explanations, formatting examples, headings, or any surrounding text.
    - Do not preface your answer with phrases like "Here is", "The values are", or anything similar.
    - Do not describe or explain your reasoning.

     Most important rule: Your entire response must consist of only the list of relevant value labels without any information.
    """

    return intro + sentence_block + annotation_block + few_shot_block + instructions


def get_few_shot_examples_per_task(sentence, annotator, task, knn_classifier, data):
    neighbors_indexes = knn_classifier.query(sentence)
    selected_examples = [data[i] for i in neighbors_indexes]

    selected_sentences = [ex['sentence'] for ex in selected_examples]
    lines = []

    selected_annotations = [ex[task][annotator] for ex in selected_examples]

    for i, sent in enumerate(selected_sentences):
        lines.append(f"Sentence: {sent}")

        ann = selected_annotations[i]
        lines.append(f"{task.capitalize()}: {ann}")

        lines.append("")

    return "\n".join(lines)


def get_few_shot_examples_all_tasks(sentence, annotator, knn_classifier, data):
    neighbors_indexes = knn_classifier.query(sentence)
    selected_examples = [data[i] for i in neighbors_indexes]

    selected_sentences = [ex['sentence'] for ex in selected_examples]
    lines = []

    selected_annotations = {
        task: [ex[task][annotator] for ex in selected_examples]
        for task in TASKS
    }

    for i, sent in enumerate(selected_sentences):
        lines.append(f"Sentence: {sent}")

        for task in TASKS:
            ann = selected_annotations[task][i]
            lines.append(f"{task.capitalize()}: {ann}")

        lines.append("")

    return "\n".join(lines)
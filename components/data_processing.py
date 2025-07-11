import pandas as pd
import json

DATA_PATH = "/home/eliscio/dev/Annotating_values/data"
TASKS = ["argument", "emotion", "sentiment", "topic", "values"]
ANNOTATORS = ["Annotator_1", "Annotator_2", "Annotator_3", "Annotator_4", "Annotator_5"]
ANNNOTATION_NUMBER = 50
SENTIMENT_MAP = {
    0: "Very negative",
    1: "Somewhat negative",
    2: "Neutral",
    3: "Somewhat positive",
    4: "Very positive"
}


def load_data():
    path = f'{DATA_PATH}/annotations_level_2.json'
    with open(path, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    return all_data


def create_annotation_dictionary():
    annotation_array = []
    initial_df = pd.read_csv(f'{DATA_PATH}/values/Annotator_1.csv')

    data_dict = {task: {ann: [] for ann in ANNOTATORS} for task in TASKS}

    # topic
    for task in TASKS:
        for annotator in ANNOTATORS:
            current_df = pd.read_csv(f'{DATA_PATH}/{task}/{annotator}.csv')

            if task == "argument":
                for i in range(ANNNOTATION_NUMBER):
                    if pd.isna(current_df['Arguments'][i + 1]):
                        current_df['Arguments'][i + 1] = "None"
                    data_dict[task][annotator].append(
                        current_df['Arguments'][i + 1]
                    )
            if task == "sentiment":
                for i in range(ANNNOTATION_NUMBER):
                    if pd.isna(current_df.iloc[i + 1, -1]):
                        current_df.iloc[i + 1, -1] = 2  # Default to Neutral if NaN
                    data_dict[task][annotator].append(
                        SENTIMENT_MAP[int(current_df.iloc[i + 1, -1])]
                    )
            else:
                for i in range(ANNNOTATION_NUMBER):
                    data_dict[task][annotator].append(
                        current_df.apply(lambda row: ', '.join(
                            [col for col in current_df.columns if row[col] == 1 and col != 'question_id']), axis=1)[
                            i + 1]
                    )

    for i, sentence in enumerate(initial_df['english'][1:ANNNOTATION_NUMBER + 1]):
        entry = {'sentence': sentence}
        for task in TASKS:
            entry[task] = {annotator: data_dict[task][annotator][i] for annotator in ANNOTATORS}
        annotation_array.append(entry)

    output_path = f'{DATA_PATH}/annotations.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(annotation_array, f, ensure_ascii=False, indent=2)
    print(f"Annotations saved to {output_path}")

    return annotation_array


def load_value_categories(path=f"{DATA_PATH}/values_level_2_mapping.json"):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def remap_values_to_level_2(in_path, out_path, mapping_path=f"{DATA_PATH}/values_level_2_mapping.json"):
    VALUE_CATEGORY = load_value_categories(mapping_path)

    with open(in_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for rec in data:
        new_vals = {}
        for ann, raw in rec.get('values', {}).items():
            cats = {
                VALUE_CATEGORY.get(v.strip(), "")
                for v in raw.split(',')
            }
            new_vals[ann] = ", ".join(sorted(cats))
        rec['values'] = new_vals

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    remap_values_to_level_2(f"{DATA_PATH}/annotations.json", f"{DATA_PATH}/annotations_level_2.json")

    # annotations = create_annotation_dictionary()
    #
    # print(annotations[:5])  # Print first 5 annotations for verification

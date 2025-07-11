import itertools
import json
from sklearn.metrics import cohen_kappa_score

EMOTIONS = [
    "admiration","amusement","anger","annoyance","approval","caring","confusion",
    "curiosity","desire","disappointment","disapproval","disgust","embarrassment",
    "excitement","fear","gratitude","grief","joy","love","nervousness","optimism",
    "pride","realization","relief","remorse","sadness","suprise"
]

SENTIMENTS = [
    "Very Negative","Somewhat Negative","Neutral","Somewhat Positive","Very Positive"
]

TOPICS = [
    "Municipality and residents engagement in the energy sector",
    "Energy storage and supplying energy in The Netherlands",
    "Wind and solar energy",
    "Market Determination Dynamics",
    "Landscapes and windmills tourism",
    "Hydrogen energy pipeline networks"
]


def compute_pairwise_kappa(data, key, categories=None):
    """
    data: list of records, each with data[key] = dict of annotator→labels
    key: "emotion" (multilabel) or "sentiment" (single-label)
    categories: list of possible labels (required for multilabel)
    returns: dict { (ann1,ann2): κ }
    """
    ann_labels = {ann: [] for ann in data[0][key]}
    for rec in data:
        for ann, lbl_str in rec[key].items():
            ann_labels[ann].append(
                [lbl.strip().lower() for lbl in lbl_str.split(",") if lbl.strip()]
            )

    results = {}
    annotators = list(ann_labels)
    for a1, a2 in itertools.combinations(annotators, 2):
        y1 = []
        y2 = []
        for i in range(len(data)):
            for cat in categories:
                cat = cat.lower()
                y1.append(1 if cat in ann_labels[a1][i] else 0)
                y2.append(1 if cat in ann_labels[a2][i] else 0)
        results[(a1, a2)] = cohen_kappa_score(y1, y2)
    return results


if __name__ == "__main__":
    with open("../data/annotations.json","r",encoding="utf-8") as f:
        data = json.load(f)

    data = data[:-1]

    # emotion
    emo_k = compute_pairwise_kappa(data, "emotion", categories=EMOTIONS)
    ranked_emo = sorted(emo_k.items(), key=lambda x: x[1], reverse=True)
    print("Emotion κ:")
    for (a1,a2), κ in ranked_emo:
        print(f"  {a1} vs {a2}  κ={κ:.3f}")

    # sentiment
    sent_k = compute_pairwise_kappa(data, "sentiment", categories=SENTIMENTS)
    ranked_sent = sorted(sent_k.items(), key=lambda x: x[1], reverse=True)
    print("\nSentiment κ:")
    for (a1, a2), κ in ranked_sent:
        print(f"  {a1} vs {a2}  κ={κ:.3f}")

    # topic
    topic_k = compute_pairwise_kappa(data, "topic", categories=TOPICS)
    ranked_topic = sorted(topic_k.items(), key=lambda x: x[1], reverse=True)
    print("\nTopic κ:")
    for (a1, a2), κ in ranked_topic:
        print(f"  {a1} vs {a2}  κ={κ:.3f}")

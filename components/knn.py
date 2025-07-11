from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors


class KNN:
    def __init__(self, k=3, model_name='all-MiniLM-L6-v2'):
        self.k = k
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.inputs = None
        self.knn = None

    def fit(self, input_annotations):
        self.inputs = input_annotations
        if self.k == 0:
            return
        self.embeddings = self.model.encode(input_annotations, convert_to_tensor=True).cpu().numpy()
        self.knn = NearestNeighbors(n_neighbors=min(self.k + 1, len(self.inputs)), metric='cosine')
        self.knn.fit(self.embeddings)

    def query(self, new_input):
        if self.k == 0:
            return []

        if self.embeddings is None:
            raise ValueError("The model has not been fitted yet.")

        new_embedding = self.model.encode([new_input], convert_to_tensor=True).cpu().numpy()
        distances, indices = self.knn.kneighbors(new_embedding)

        # Flatten the arrays
        indices = indices[0]

        is_duplicate = new_input in self.inputs

        # Filter out the index where input matches new_input
        filtered_indices = [
                               idx for idx in indices if self.inputs[idx] != new_input
                           ][:self.k]
        if is_duplicate:
            return filtered_indices[:self.k - 1]
        else:
            return filtered_indices[:self.k]


def main():
    training_inputs = [
        "Translate English to French: How are you?",
        "Translate English to Spanish: Good morning.",
        "What is the capital of France?",
        "Summarize the article in one sentence."
    ]

    knn_classifier = KNN(k=2)

    knn_classifier.fit(training_inputs)

    query_input = "Translate English to French: Good evening."
    closest_idx = knn_classifier.query(query_input)
    print(closest_idx)


if __name__ == "__main__":
    main()

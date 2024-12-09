import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import pickle  # For saving and loading the model

class ThreeLayerRBM:
    def __init__(self, n_visible, n_hidden_middle, n_hidden_top):
        self.n_visible = n_visible
        self.n_hidden_middle = n_hidden_middle
        self.n_hidden_top = n_hidden_top

        # Initialize weights and biases
        self.W_vm = np.random.normal(0, 0.01, size=(self.n_visible, self.n_hidden_middle))
        self.W_mt = np.random.normal(0, 0.01, size=(self.n_hidden_middle, self.n_hidden_top))

        self.b_v = np.zeros(self.n_visible)
        self.b_m = np.zeros(self.n_hidden_middle)
        self.b_t = np.zeros(self.n_hidden_top)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sample_prob(self, probs):
        return np.random.binomial(n=1, p=probs)

    def train(self, data_v, data_h, learning_rate=0.01, k=1, epochs=100, batch_size=10, l1_reg=0.01, l2_reg=0.01):
        n_samples = data_v.shape[0]
        for epoch in range(epochs):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            data_v = data_v[indices]
            data_h = data_h[indices]

            for batch_start in range(0, n_samples, batch_size):
                v0 = data_v[batch_start:batch_start + batch_size]
                h_t_labels = data_h[batch_start:batch_start + batch_size]

                # Positive phase
                p_hm_v0 = self.sigmoid(np.dot(v0, self.W_vm) + self.b_m)
                h_m0 = self.sample_prob(p_hm_v0)
                h_t0 = h_t_labels

                # Negative phase
                for _ in range(k):
                    p_hm_ht0 = self.sigmoid(np.dot(h_t0, self.W_mt.T) + self.b_m)
                    h_mk = self.sample_prob(p_hm_ht0)
                    p_v_hmk = self.sigmoid(np.dot(h_mk, self.W_vm.T) + self.b_v)
                    vk = self.sample_prob(p_v_hmk)
                    p_hm_vk = self.sigmoid(np.dot(vk, self.W_vm) + self.b_m)
                    h_mk = self.sample_prob(p_hm_vk)
                    p_ht_hmk = self.sigmoid(np.dot(h_mk, self.W_mt) + self.b_t)
                    h_tk = self.sample_prob(p_ht_hmk)

                dW_vm = np.dot(v0.T, p_hm_v0) - np.dot(vk.T, p_hm_vk)
                dW_mt = np.dot(p_hm_v0.T, h_t0) - np.dot(p_hm_vk.T, h_tk)
                db_v = np.sum(v0 - vk, axis=0).astype(np.float64)
                db_m = np.sum(p_hm_v0 - p_hm_vk, axis=0)
                db_t = np.sum(h_t0 - h_tk, axis=0).astype(np.float64)

                # Apply regularization
                dW_vm -= l2_reg * self.W_vm
                dW_mt -= l2_reg * self.W_mt
                db_t -= l1_reg * np.sign(self.b_t)  # Sparsity in top layer

                # Update parameters
                self.W_vm += learning_rate * dW_vm / batch_size
                self.W_mt += learning_rate * dW_mt / batch_size
                self.b_v += learning_rate * db_v / batch_size
                self.b_m += learning_rate * db_m / batch_size
                self.b_t += learning_rate * db_t / batch_size

            if (epoch + 1) % 10 == 0:
                error = np.mean((v0 - vk) ** 2)
                print(f'Epoch {epoch + 1}, Reconstruction Error: {error}')

    def transform(self, v):
        # Forward pass to top layer
        p_hm_v = self.sigmoid(np.dot(v, self.W_vm) + self.b_m)
        h_m = self.sample_prob(p_hm_v)
        p_ht_hm = self.sigmoid(np.dot(h_m, self.W_mt) + self.b_t)
        h_t = self.sample_prob(p_ht_hm)
        return h_t

    def save_model(self, filename):
        model_params = {
            'W_vm': self.W_vm,
            'W_mt': self.W_mt,
            'b_v': self.b_v,
            'b_m': self.b_m,
            'b_t': self.b_t,
            'n_visible': self.n_visible,
            'n_hidden_middle': self.n_hidden_middle,
            'n_hidden_top': self.n_hidden_top
        }
        with open(filename, 'wb') as f:
            pickle.dump(model_params, f)
        print(f'Model saved to {filename}')

    def load_model(self, filename):
        with open(filename, 'rb') as f:
            model_params = pickle.load(f)
        self.W_vm = model_params['W_vm']
        self.W_mt = model_params['W_mt']
        self.b_v = model_params['b_v']
        self.b_m = model_params['b_m']
        self.b_t = model_params['b_t']
        self.n_visible = model_params['n_visible']
        self.n_hidden_middle = model_params['n_hidden_middle']
        self.n_hidden_top = model_params['n_hidden_top']
        print(f'Model loaded from {filename}')


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def extract_features(df, findings_list, hidden_features_dict):
    df['text'] = df['FINDINGS'].fillna('') + ' ' + df['IMPRESSION'].fillna('')
    df['clean_text'] = df['text'].apply(preprocess_text)

    vectorizer_v = CountVectorizer(vocabulary=findings_list, binary=True)
    X_visible = vectorizer_v.fit_transform(df['clean_text']).toarray()

    hidden_features = []
    hidden_vectorizers = {}
    for category, terms in hidden_features_dict.items():
        vectorizer_h = CountVectorizer(vocabulary=terms, binary=True)
        X_hidden = vectorizer_h.fit_transform(df['clean_text']).toarray()
        hidden_features.append(X_hidden)
        hidden_vectorizers[category] = vectorizer_h
    X_hidden = np.concatenate(hidden_features, axis=1)
    return X_visible, X_hidden, vectorizer_v, hidden_vectorizers

# Usage Example (same as original code)
data_file = 'output.csv'
df = pd.read_csv(data_file)

findings_list = [
    "left", "right", "atelectasis", "bronchiectasis", "bulla", "consolidation", "dextrocardia", "effusion", "emphysema",
    "fracture clavicle", "fracture rib", "groundglass opacity", "interstitial opacification",
    "mass paraspinal", "mass soft tissue", "nodule", "opacity", "pneumomediastinum", "pneumonia",
    "pneumoperitoneum", "pneumothorax", "pleural effusion", "pulmonary edema", "scoliosis",
    "tuberculosis", "volume loss", "rib", "mass", "infiltration", "other findings"
]

hidden_features_dict = {
    'location': [
        "left lung", "right lung", "upper lobe", "lower lobe", "cardiac region",
        "pleural space", "diaphragm", "mediastinum", "thoracic spine", "abdominal region"
    ],
    'organ_system': [
        "respiratory system", "cardiovascular system", "musculoskeletal system", "digestive system"
    ],
    'mode_of_pathology': [
        "congenital", "acquired", "infection", "inflammation", "tumor", "degenerative", "vascular"
    ],
    'severity': [
        "mild", "moderate", "severe",
    ],
}

if __name__ == '__main__':
    X_visible, X_hidden, vectorizer_v, hidden_vectorizers = extract_features(df, findings_list, hidden_features_dict)

    n_visible = X_visible.shape[1]
    n_hidden_middle = 30
    n_hidden_top = X_hidden.shape[1]

    rbm = ThreeLayerRBM(n_visible, n_hidden_middle, n_hidden_top)
    rbm.train(X_visible, X_hidden, learning_rate=0.05, epochs=500, batch_size=10)

    rbm.save_model('rbm_model.pkl')
    with open('vectorizer_v.pkl', 'wb') as f:
        pickle.dump(vectorizer_v, f)
    with open('hidden_vectorizers.pkl', 'wb') as f:
        pickle.dump(hidden_vectorizers, f)

    # Define a helper function to get cond_z
    def get_cond_z(rbm, X_visible_batch):
        # This returns the top-layer activations, which serve as cond_z
        cond_z = rbm.transform(X_visible_batch)
        return cond_z

    # Example of extracting cond_z from new data (using the first 5 samples)
    new_data_v = X_visible[:5]
    cond_z = get_cond_z(rbm, new_data_v)
    print("cond_z for first 5 samples:")
    print(cond_z)
import pickle

with open("face_data/master_embeddings.pkl", "rb") as f:
    data = pickle.load(f)
print(data)

import pickle

with open("./wiki/single_data_label_description_1975_1980.pkl", "rb") as file:
    data = pickle.load(file)
print(data)
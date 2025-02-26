import pandas as pd
import os

data = pd.read_csv("C:/Users/019072/Downloads/heart.csv")

path=os.path.join("data","raw")
os.makedirs(path)

data.to_csv(os.path.join(path,"data.csv"),index=False)
from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
import pickle

category_data = pd.read_csv('idx2category.csv')
idx2category = {row.k: row.v for idx, row in category_data.iterrows()}

with open("rdmf.pickle", mode='rb') as f:
    model = pickle.load(f)

def index(request):
    if request.method == "GET":
        return render(
            request,
            "nlp/home.html"
        )
    else:
        title = [request.POST["title"]]
        result = model.predict(title)[0]
        print("title:",title)
        pred = idx2category[result]
        print("result:",result)
        return render(
            request,
            "nlp/home.html",
            {"title": pred}
        )
        


    

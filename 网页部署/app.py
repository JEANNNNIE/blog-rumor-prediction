from flask import Flask,request, url_for, redirect, render_template
from flask_ngrok import run_with_ngrok
import pickle
import jieba
import numpy as np

app = Flask(__name__)
run_with_ngrok(app)

model = pickle.load(open('random_forest.pkl','rb')) # 随机森林模型
vectorizer = pickle.load(open('vectorizer.pkl','rb')) # 向量构建词典（包含去停用词）
transformer = pickle.load(open('TfidfTransformer.pkl','rb')) # tfidf


@app.route('/')
def hello_world():
    return render_template("rumor.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    print("0----------------------------------------------------------")
    txt = request.values.get("Blog") 
    print("1----------------------------------------------------------")
    test_depart = " ".join(jieba.cut(txt))
    predict = model.predict(transformer.transform(vectorizer.transform([test_depart])))
    output = int(predict[0])

    if output>0.5:
        return render_template('rumor.html',pred='This is a rumor probably.',bhai="kuch karna hain iska ab?")
    else:
        return render_template('rumor.html',pred='It is likely to be true.',bhai="Your Forest is Safe for now")


if __name__ == '__main__':
    app.run()

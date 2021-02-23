from numpy import*
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import jieba
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def seg_depart(sentence):
    # jieba分词
    sentence_depart = jieba.cut(sentence)
    outstr = " ".join(sentence_depart)
    # 创建停用词表
    #stopwords = stopwordslist()
    # 输出
    # outstr = []
    #outstr = ""
    #for word in sentence_depart:
    #    if word not in stopwords:
    #        outstr += word
    #        outstr += " "

    return outstr


# 导入停用词表
def stopwordslist():
    stopfile = open("stopwords.txt", encoding="utf-8", errors='ignore').readlines()
    stopwords = list()
    for word in stopfile:
        stopwords.append(word.strip())

    return stopwords


# 计算tf_idf
def tf_idf(full_set, stopwords):
    vectorizer = CountVectorizer(min_df=1e-4, stop_words=stopwords)  # drop df < 1e-5,去低频词
    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(vectorizer.fit_transform(full_set))
    pickle.dump(vectorizer,open('vectorizer.pkl','wb'))
    print("字典保存成功")
    pickle.dump(transformer,open('TfidfTransformer.pkl','wb'))
    print("tf-idf转换器保存成功")
    words = vectorizer.get_feature_names()
    print("how many words: {0}".format(len(words)))
    print("tf-idf shape: ({0},{1})".format(tfidf.shape[0], tfidf.shape[1]))
    return tfidf


def main():
    f1 = open("total_data.csv", encoding="utf-8", errors='ignore')

    t_data = pd.read_csv(f1,sep=" ")

    full_list = []
    labellist = []

    length1 = len(t_data)
    print("共" + str(length1) + "条数据")

    stopwords = stopwordslist()
    for i in range(length1):
        wordlist = seg_depart(t_data.iloc[i, 1])
        full_list.append(wordlist)
        labellist.append(t_data.iloc[i, 2])
    print(1)
    print(full_list[0])
    print(labellist[0])
    full = tf_idf(full_list, stopwords)
    print(type(full))
    print(1.5)
    full = full.toarray()

    print(2)
    # 分层划分训练集和测试集，比例8:2
    train_set, test_set, train_label, test_label = train_test_split(full, labellist, test_size=0.2, stratify=labellist)
    print("-------数据集划分--------")
    print(str(len(train_set)) + "--" + str(len(train_label)))
    print(str(len(test_set)) + "--" + str(len(test_label)))

    # 朴素贝叶斯94.29%
    classifier = MultinomialNB().fit(train_set, train_label)
    pickle.dump(classifier,open('bayes.pkl','wb'))
    print("朴素贝叶斯模型保存成功")
    y_pred1 = classifier.predict(test_set)
    print("val mean accuracy of Bayes: {0}".format(classifier.score(test_set, test_label)))
    print(classification_report(test_label, y_pred1))

    # 逻辑回归94.54%
    lr_model = LogisticRegression()
    lr_model.fit(train_set, train_label)
    pickle.dump(lr_model,open('logistic.pkl','wb'))
    print("逻辑回归模型保存成功")
    print("val mean accuracy of logistic regression: {0}".format(lr_model.score(test_set, test_label)))
    y_pred = lr_model.predict(test_set)
    print(classification_report(test_label, y_pred))

    # 随机森林 95.51%（10） 95.68%（15）
    clf = RandomForestClassifier(n_estimators=15)
    clf.fit(train_set, train_label)
    pickle.dump(clf,open('random_forest.pkl','wb'))
    print("随机森林模型保存成功")
    print("val mean accuracy of random forest: {0}".format(clf.score(test_set, test_label)))
    y_pred2 = clf.predict(test_set)
    print(classification_report(test_label, y_pred2))


if __name__ == '__main__':
    main()

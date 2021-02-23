import pickle
import jieba

model = pickle.load(open('random_forest.pkl','rb')) # 随机森林模型
vectorizer = pickle.load(open('vectorizer.pkl','rb')) # 向量构建词典（包含去停用词）
transformer = pickle.load(open('TfidfTransformer.pkl','rb')) # tfidf

test_str = "红十字会路虎一排排，权威人士爆料：位于北京北新桥三条8号的中国红十字总会的车库里，停着若干豪华公车。红会司局级以上领导，每人两辆。此举是为应付北京限行而为。本人对此未作调查，但结合最近爆出的红色黑幕，不管你信不信，反正我是信了。消息报料人，正是9年前揭开希望工程黑幕的一代名记方进玉 ' >  "
test_depart = " ".join(jieba.cut(test_str))
predict = model.predict(transformer.transform(vectorizer.transform([test_depart])))
print(predict) # 1-谣言；0-正常
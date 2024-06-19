# 导入必要的库
import re
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# 定义一个基础的问题-答案存储类
class QAPairs:
    def __init__(self):
        self.qa_pairs = {}

    def add_pair(self, question, answer):
        self.qa_pairs[question.lower()] = answer

    def get_answer(self, question):
        question = question.lower()
        if question in self.qa_pairs:
            return self.qa_pairs[question]
        else:
            return "抱歉，我暂时无法回答这个问题。"


# 使用预训练的模型，如BERT进行文本分类
class TextClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    def classify(self, text, candidate_labels):
        labels = self.classifier(text, candidate_labels)
        return labels['labels'][0]


# 定义一个基础的问答系统类
class QASystem:
    def __init__(self):
        self.qa_pairs = QAPairs()
        self.classifier = TextClassifier()

        # 添加预定义的问题和答案
        self.qa_pairs.add_pair("什么是基金？",
                               "基金是一种集合投资方式，由多位投资者共同出资形成的资金池，由专业的基金管理人按照既定的投资策略进行投资。")
        self.qa_pairs.add_pair("A股是什么？",
                               "A股是中国内地上市公司的股票市场，是指在上海证券交易所和深圳证券交易所上市交易的股票。")
        self.qa_pairs.add_pair("如何查询基金的持仓明细？",
                               "你可以通过基金公司的官方网站或者证券交易所的信息披露平台查询基金的持仓明细。")
        self.qa_pairs.add_pair("如何查询A股的日行情？", "你可以通过证券交易所的官方网站或者证券行情软件查询A股的日行情。")
        self.qa_pairs.add_pair("如何处理多表之间的复杂关联？",
                               "处理多表关联可以使用SQL语句中的JOIN操作来实现，通过指定连接条件将多个表中的数据关联起来。")
        self.qa_pairs.add_pair("如何处理长文本的复杂结构？",
                               "处理长文本可以使用自然语言处理工具进行分词、实体识别和关键信息抽取，以提取文本中的关键信息。")
        self.qa_pairs.add_pair("如何分块处理超长文本？",
                               "可以将超长文本分成适当大小的块，然后分别处理每个文本块，最后合并结果。")
        self.qa_pairs.add_pair("如何创建一个问答系统？",
                               "可以使用大语言模型，通过预定义的问题和答案进行问答，也可以结合实体识别和文本分类等技术来提升系统的智能程度。")

    def add_qa_pair(self, question, answer):
        self.qa_pairs.add_pair(question, answer)

    def answer_question(self, question):
        return self.qa_pairs.get_answer(question)

    def classify_question(self, question):
        labels = self.classifier.classify(question, list(self.qa_pairs.qa_pairs.keys()))
        return labels

    def process_long_text(self, text):
        # 省略具体的长文本处理方法，可以根据需求扩展
        pass


# 测试问答系统
if __name__ == "__main__":
    qa_system = QASystem()

    while True:
        user_input = input("请输入你的问题（输入'退出'结束程序）：")
        if user_input.lower() == "退出":
            print("谢谢使用！再见！")
            break
        else:
            # 先进行文本分类，确定问题类型
            predicted_label = qa_system.classify_question(user_input)

            # 如果问题属于预定义问题集合中的某一个，直接返回答案
            if predicted_label in qa_system.qa_pairs.qa_pairs:
                answer = qa_system.answer_question(predicted_label)
                print("答案：", answer)
            else:
                print("抱歉，我暂时无法回答这个问题。请尝试其它问题。")

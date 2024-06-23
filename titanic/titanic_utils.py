import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

def print_model_accuracy(model, X_train, y_train, X_test):
    """ 给定一个模型，输出它在训练数据和测试数据上的准确率 """
    # 训练数据上的准确率
    predictions = model.predict(X_train)
    correct_predictions = predictions == y_train
    train_accuracy = correct_predictions.mean()
    print(f'训练数据集准确率: {train_accuracy:.2%}')

    # # 训练数据上交叉验证的准确率
    # cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    # cv_accuracy = cv_scores.mean()
    # print(f'训练数据交叉验证准确率: {cv_accuracy:.2%}')

    test_accuracy = evaluate_test_accuracy(model, X_test)
    print(f'测试数据集准确率: {test_accuracy:.2%}')


def evaluate_test_accuracy(model, X_test):
    """ 给定一个模型，返回它在测试数据上的准确率 """
    # 注意：这里我们使用的是带有y标签的测试数据，这样我们可以直接在本地就能评估模型的测试准确率。
    # 在实际的Kaggle比赛中，比赛提供的测试数据显然不带y标签，我们需要把模型预测的结果提交到Kaggle上，
    # 然后Kaggle会给出测试数据的准确率。在这里因为Titanic题目是判断乘客是否幸存，实际上幸存者名
    # 单是公开信息，所以我们可以自行构造带y标签的测试数据。看到Kaggle上有很多人提交了测试准确率100%
    # 的结果，也是这个原因。
    test_df = pd.read_csv('./data/test_with_y.csv')
    y_test = test_df['Survived']

    predictions = model.predict(X_test)
    correct_predictions = predictions == y_test
    test_accuracy = correct_predictions.mean()
    predictions = model.predict(X_test)

    correct_predictions = predictions == y_test
    test_accuracy = correct_predictions.mean()
    return test_accuracy

def evaluate_submission_accuracy(submission_file):
    """ 给定一个提交文件，返回它在测试数据上的准确率 """
    test_df = pd.read_csv('./data/test_with_y.csv')
    y_test = test_df['Survived']

    predictions = pd.read_csv(submission_file)['Survived']
    accuracy = accuracy_score(y_test, predictions)
    return accuracy

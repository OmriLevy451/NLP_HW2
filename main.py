import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectKBest, f_regression, f_classif


def classify(train_file, test_file):
    # todo: implement this function
    print(f'starting feature extraction and classification, train data: {train_file} and test: {test_file}')

    train_data=[]
    test_data=[]
    target=[]
    test_target=[]

    with open(train_file, 'r') as file:
        for line in file:
            review=json.loads(line)
            train_data.append(review.get('reviewText','summary'))
            target.append(review['overall'])

    with open(test_file, 'r') as file2:
        for line in file2:
            Treview=json.loads(line)

            test_data.append(Treview.get('reviewText', 'summary'))
            test_target.append(Treview['overall'])


    text_clf=Pipeline([('vectorizer',CountVectorizer(ngram_range=(1,3), lowercase=True, stop_words='english', max_features=5000, min_df=1, max_df=0.95))
                          ,('tfidf',TfidfTransformer(use_idf=True, smooth_idf=True, sublinear_tf=True)),
                       ('clf',LogisticRegression(solver='newton-cg', penalty='l2', class_weight='balanced',max_iter=5000 ,C=1))])



    param_grid = {

    }

    grid_search = GridSearchCV(
        text_clf,
        param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1
    )
    grid_search.fit(train_data, target)
    predictions = grid_search.predict(test_data)

    vectorizer=CountVectorizer(ngram_range=(1, 2), lowercase=True, stop_words='english', max_features=5000, min_df=2, max_df=0.95)
    vec_train_data=vectorizer.fit_transform(train_data)

    selector=SelectKBest(score_func=f_classif, k=15)
    selector.fit_transform(vec_train_data, target)
    selected_features = selector.get_support(indices=True)
    feature_names = np.array(vectorizer.get_feature_names_out())[selected_features]

    print(feature_names)

    print("Best hyperparameters:", grid_search.best_params_)


    test_results = {'class_1_F1': 0.0,
                    'class_2_F1': 0.0,
                    'class_3_F1': 0.0,
                    'class_4_F1': 0.0,
                    'class_5_F1': 0.0,
                    'accuracy': 0.0}



    unique_classes = sorted(set(target))
    for class_label in unique_classes:
        f1 = f1_score(test_target, predictions, labels=[class_label], average=None)
        test_results[f'class_{int(class_label)}_F1'] = f1[0] if len(f1) > 0 else 0.0
    test_results['accuracy']=accuracy_score(test_target, predictions)

    confmet=confusion_matrix(test_target, predictions, labels=[1,2,3,4,5])
    "The classes with the most confusion are those with a wider distribution of values over all classes instead of a large spike in instances"
    "Classes 3 and 4 suffer from this the most."
    "We can see that class 4 predicted class 4 cases as class 5 nearly as much as it correctly did class 4 itself."

    "Same with class 3 in which the sum of its predictions for class 1 and 2 outweighs its predictions for class 3."
    "something which does not occur in class 1 for example in which the sum of ALL other classes does not overshadow its true prediction for class one"
    "A much higher result in a single class shows clear lack of confusion."



    "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"


    print(confmet)


    return test_results


if __name__ == '__main__':
    with open('config.json', 'r') as json_file:
        config = json.load(json_file)

    results = classify(config['train_data'], config['test_data'])

    for k, v in results.items():
        print(k, v)

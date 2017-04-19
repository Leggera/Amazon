import pandas as pd
import gensim
from gensim.models import Doc2Vec
from sklearn.linear_model import LogisticRegression as LogReg
#from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from time import time
import os
from sklearn.metrics import classification_report
import numpy as np
#import statsmodels.api as sm

def DocumentVectors(model, model_name):
    if (model_name == "word2vec_c"):
        model_w2v = gensim.models.Doc2Vec.load_word2vec_format(model , binary=False)
        vec_vocab = [w for w in model_w2v.vocab if "_*" in w]
        vec_vocab = sorted(vec_vocab, key = lambda x: int(x[2:]))
        DocumentVectors0 = [model_w2v[w] for w in vec_vocab[:25000]]
        DocumentVectors1 = [model_w2v[w] for w in vec_vocab[25000:50000]]
    elif(model_name == "doc2vec"): #TODO
        
        try:
            model_d2v = Doc2Vec.load(model)
            
            
        except:
            print ("trouble loading model")
            print (model)
            exit()
        #print model_d2v.docvecs.doctags
        DocumentVectors0 = np.array([model_d2v.docvecs[tag] for tag in model_d2v.docvecs.doctags if 'train' in tag])
        DocumentVectors1 = np.array([model_d2v.docvecs[tag] for tag in model_d2v.docvecs.doctags if 'test' in tag])
        train_labels = [tag.split()[2] for tag in model_d2v.docvecs.doctags if 'train' in tag] #TODO int(split)? 
        test_labels = [tag.split()[2] for tag in model_d2v.docvecs.doctags if 'test' in tag] 
    return (DocumentVectors0, train_labels, DocumentVectors1, test_labels)

def Classification(classifier, train, train_labels, test, test_labels):
    
    grid_search = GridSearchCV(classifiers_dict[classifier], param_grid = search_parameters[classifier], error_score=0.0, n_jobs = -1)
    
    t0 = time()
    grid_search.fit(train, train_labels)
    print("done in %0.3fs" % (time() - t0))
    #print("Best score: %0.3f" % grid_search.best_score_)
    best_parameters = grid_search.best_estimator_.get_params()
    k = ""
    for param_name in sorted(search_parameters[classifier].keys()):
        #print("%s: %r" % (param_name, best_parameters[param_name]))
        k += "%s: %r\n" % (param_name, best_parameters[param_name])
    test_prediction = grid_search.predict(test)
    test_scores = (classification_report(test_labels, test_prediction)).split('\n')#TODO .2f -> .3f
    test_score =  ' '.join(test_scores[0].lstrip().split(' ')[:-1]) +'\n' + ' '.join(test_scores[-2].split(' ')[3:-1])
    train_prediction = grid_search.predict(train)
    train_accuracy = sum(train_prediction == train_labels)/len(train_labels)
    train_scores = (classification_report(train_labels, train_prediction)).split('\n')
    train_score =  ' '.join(train_scores[0].lstrip().split(' ')[:-1]) +'\n' + ' '.join(test_scores[-2].split(' ')[3:-1])
    test_accuracy = sum(test_prediction == test_labels)/len(test_labels)
    return 'cv %.3f test %.3f train %.3f' % (grid_search.best_score_, test_accuracy, train_accuracy) + '\n' + 'train: ' + train_score + '\n' + 'test: ' + test_score, k[:-1]
    
    '''train_prediction = classifier.predict(train)
    train_accuracy = (100 * float(len(train_prediction[train_prediction == train_labels]))/len(train_labels))
    test_prediction = classifier.predict(test)
    test_accuracy = (100 * float(len(test_prediction[test_prediction == test_labels]))/len(test_labels))
    return 'train %.3f/test %.3f' % (train_accuracy, test_accuracy)'''
if __name__ == "__main__":
    min_count = 1# 5#TODO
    threads = 24# 20, 1#TODO
    
    d0 = ['implementation']
    parameters = ['size', 'alpha', 'window', 'negative']
    columns = ['size', 'alpha', 'window', 'negative', 'cbow0_sample', 'cbow1_sample']
    min_c= ['min_count']#TODO
    classifiers = ['SklearnLogReg', 'SklearnLinearSVC']#, 'SklearnMLP'
    d3 = ['threads']
    best_params = ['best_parameters']
    df= pd.DataFrame(columns = d0+columns+min_c+classifiers+best_params + d3)
    
    default_parameters = dict()
    classifiers_dict=dict()
    search_parameters = dict()
    space_dir = dict()
    
    default_parameters['size'] = 150
    default_parameters['alpha'] = 0.05
    default_parameters['window'] = 10
    default_parameters['negative'] = 25
    
    classifiers_dict['SklearnLogReg'] = LogReg()
    #classifiers_dict['SklearnMLP'] = MLPClassifier(hidden_layer_sizes = (50, 50), max_iter=1000)
    classifiers_dict['SklearnLinearSVC'] = LinearSVC()
    #classifiers_dict['StatModelsLogReg'] = sm.Logit()

    search_parameters['SklearnLogReg'] = {'solver' : ('newton-cg', 'lbfgs', 'liblinear', 'sag'), 'penalty': ('l1', 'l2'), 'dual': (False, True), 'fit_intercept': (True, False), 'intercept_scaling': (1, 2, 3), 'max_iter': (100, 200, 400, 800, 1000), 'multi_class': ('ovr', 'multinomial')}
    #search_parameters['SklearnMLP'] = {'solver' : ('lbfgs', 'sgd', 'adam')}#TODO
    search_parameters['SklearnLinearSVC'] = {'loss' : ('hinge', 'squared_hinge'), 'penalty': ('l1', 'l2'), 'dual': (False, True), 'fit_intercept': (True, False), 'intercept_scaling': (1, 2, 3),  'max_iter': (100, 200, 400, 800, 1000), 'multi_class': ('ovr', 'crammer_singer')}
    
    space_dir["word2vec_c"] = "space_w2v_20ng/"
    space_dir["doc2vec"] = "diploma/"
    
    index  = 0
    for model_name in [ "doc2vec"]: #TODO
        for model in os.listdir(space_dir[model_name]):
            if model.endswith('.txt'):
                if ('cbow 0' in model):
                    consider = True
                    par_list = []
                    string = model.split(".txt")[0]
                    implementation = string.split()[0]
                    
                    for column in parameters:    
                        i = string.find(column)

                        if (i != -1):
                            value = string[i:].split()[1]
                            par_list += [column + ' ' + value]
                        else:
                            par_list += [column + ' -1']
                    if (not consider):
                        continue

                    for other_model in os.listdir(space_dir[model_name]):
                        if other_model.endswith('.txt'):
                            if ('cbow 1' in other_model):
                                consider = True
                                other_model = other_model.split(".txt")[0]
                                for column in parameters:
                                    i = other_model.find(column)
                        
                                    if (i != -1):
                                        if (column + ' ' + other_model[i:].split()[1]) not in par_list:
                                            consider = False
                                            break
                                    else:
                                        if (column + ' -1') not in par_list:
                                            consider = False
                                            break
                                if (not consider):
                                    continue
                                index += 1
                                    
                                for column in parameters:    
                                    i = string.find(column)

                                    if (i != -1):
                                        value = string[i:].split()[1]
                                        df.set_value(index, column, value)
                                    else:
                                        df.set_value(index, column, default_parameters[column])
                                
                                i = string.find('sample')
                                if (i != -1):
                                    value = string[i:].split()[1]
                                    df.set_value(index, 'cbow0_sample', value)
                                else:
                                    df.set_value(index, 'cbow0_sample', '1e-2')

                                

                                i = other_model.find('sample')
                                if (i != -1):
                                    value = other_model[i:].split()[1]
                                    df.set_value(index, 'cbow1_sample', value)
                                else:
                                    df.set_value(index, 'cbow1_sample', '1e-4')

                                DocumentVectors0_0, train_labels, DocumentVectors1_0, test_labels = DocumentVectors(space_dir[model_name]+model, model_name)
                                DocumentVectors0_1, train_labels, DocumentVectors1_1, test_labels = DocumentVectors(space_dir[model_name]+other_model+'.txt', model_name)
                                try:
                                    DocumentVectors0 = np.concatenate((DocumentVectors0_0, DocumentVectors0_1), axis=1)
                                except:
                                    print (DocumentVectors0_0.shape)
                                    print (DocumentVectors0_1.shape)
                                    print (DocumentVectors1_0.shape)
                                    print (DocumentVectors1_1.shape)
                                    print (model)
                                    print (other_model)
                                DocumentVectors1 = np.concatenate((DocumentVectors1_0, DocumentVectors1_1), axis=1)

                                for classifier in classifiers:
                                    try:
                                        accuracy, best = Classification(classifier, DocumentVectors0, train_labels, DocumentVectors1, test_labels)
                                    except:
                                        print ("classification problem")
                                        print (model)
                                        print (other_model)
                                        print (DocumentVectors0_0.shape)
                                        print (DocumentVectors0_1.shape)
                                        print (DocumentVectors1_0.shape)
                                        print (DocumentVectors1_1.shape)
                                        exit()
                                    df.set_value(index, classifier, accuracy)
                                    df.set_value(index, 'best_parameters', best)
                    df.to_csv("Results_concat_20ng_second.csv")
                    print (model)

    df.to_csv("Results_concat_20ng_second.csv")

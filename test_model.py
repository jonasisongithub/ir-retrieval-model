import pandas as pd
import pyterrier as pt

def test_model(model):
    topics = pd.DataFrame([
        {'qid': '1', 'query': 'before'},
        {'qid': '2', 'query': 'css before'},
        {'qid': '3', 'query': 'after'},
        {'qid': '4', 'query': 'CSS after'},
    ])

    qrels = pd.DataFrame([
        {'qid': '1', 'docno': 'd1', 'relevance': 1}, #d1 is the only relevant document for query 1
        {'qid': '2', 'docno': 'd1', 'relevance': 1}, #d1 is the only relevant document for query 2
        {'qid': '3', 'docno': 'd2', 'relevance': 1}, #d2 is the only relevant document for query 3
        {'qid': '3', 'docno': 'd2', 'relevance': 1}, #d2 is the only relevant document for query 3
    ])

    print("Overall performance:\n")
    print(pt.Experiment([model], topics, qrels, eval_metrics=['ndcg_cut_3', 'P_1']))
    print("\n")

    print("Single word search:\n")
    print(model.search("sport"))    
    print("\n")
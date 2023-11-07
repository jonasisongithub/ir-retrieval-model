# load data

from tira.third_party_integrations import ir_datasets, ensure_pyterrier_is_loaded
import pyterrier as pt
if not pt.started():
    pt.init()

ensure_pyterrier_is_loaded()

training_dataset = 'ir-lab-jena-leipzig-wise-2023/training-20231104-training'
validation_dataset = 'ir-lab-jena-leipzig-wise-2023/validation-20231104-training'

dataset = ir_datasets.load(training_dataset)
documents = dataset.docs_iter()

# create index
def create_index(documents):
    indexer = pt.IterDictIndexer("./tmp/index", overwrite=True, meta={'docno': 100, 'text': 20480})
    index_ref = indexer.index(({'docno': i.doc_id, 'text': i.text} for i in documents))
    return pt.IndexFactory.of(index_ref)

index = create_index(documents)

# create model

bm25 = pt.BatchRetrieve(index, wmodel="BM25")

# test

#### overall performance
# The information needs that we want to test
import pandas as pd

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

pt.Experiment([bm25], topics, qrels, eval_metrics=['ndcg_cut_3', 'P_1'])

#### single word
bm25.search("sport")

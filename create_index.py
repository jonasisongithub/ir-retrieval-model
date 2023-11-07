import pyterrier as pt

def create_index(documents):
    indexer = pt.IterDictIndexer("C:/Users/Jonas/IR", overwrite=True, meta={'docno': 100, 'text': 20480})
    index_ref = indexer.index(({'docno': i.doc_id, 'text': i.text} for i in documents))
    return pt.IndexFactory.of(index_ref)

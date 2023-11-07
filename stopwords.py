from tira.third_party_integrations import ir_datasets

def load_dataset():
    training_dataset = 'ir-lab-jena-leipzig-wise-2023/training-20231104-training'
    validation_dataset = 'ir-lab-jena-leipzig-wise-2023/validation-20231104-training'

    dataset = ir_datasets.load(training_dataset)
    docs_iteration = dataset.docs_iter()

    docs_list = list(docs_iteration)

    if len(docs_list) >= 5:
        last_5_docs = docs_list[-5:]
        for i, doc in enumerate(last_5_docs):
            print(f"Letzter Datensatz {i+1}: {doc}")
   
load_dataset()



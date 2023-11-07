from tira.third_party_integrations import ir_datasets

def load_dataset():
    training_dataset = 'ir-lab-jena-leipzig-wise-2023/training-20231104-training'
    validation_dataset = 'ir-lab-jena-leipzig-wise-2023/validation-20231104-training'

    dataset = ir_datasets.load(training_dataset)
    return dataset.docs_iter()
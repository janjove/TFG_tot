import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pipilenes import NetejaNAColumns, OmpleNans, LabelEncoding
from sklearn.pipeline import Pipeline
from func_sel import filter_significant_features

def dataset_dif2(df, dict, df_variables):
    # Inicialització dels datasets
    def initialize_datasets(df):
        datasets = {}
        dataset_names = [
            'IQ', 'SAVRY', 'VAST', 'PCL','PCLYVM', 'CAPE', 'YPI', 'RPQ', 'CCA', 'SD3', 
            'ICUJ', 'ICUT', 'TRFM', 'YSR', 'TriPM', 'TRFT','DD','TRFT DMS'
        ]
        
        for name in dataset_names:
            datasets[name] = pd.DataFrame(index=df.index)
            datasets[name]['temps_fins_reincidencia1a'] = df['temps_fins_reincidencia1a'].apply(lambda x: 1095 if x > 1095 else x)
        
        return datasets

    datasets = initialize_datasets(df)

    # Mapeig de categories a datasets
    category_map = {
        'Test IQ': 'IQ', 'SAVRY': 'SAVRY','DD':'DD' ,'VAST': 'VAST', 'YPI': 'YPI',
        'RPQ': 'RPQ', 'CCA': 'CCA', 'SD3': 'SD3', 'ICUJ': 'ICUJ',
        'ICUT': 'ICUT', 'TRFM': 'TRFM', 'YSR': 'YSR', 'TriPM': 'TriPM',
        'TRFT': 'TRFT', 'TRFT DMS': 'TRFT DMS', 'CAPe': 'CAPE', 'CAPo': 'CAPE',
        'PCLe': 'PCL', 'PCLo': 'PCL', 'PCLj': 'PCL', 'PCLx': 'PCL', 'PCLYVM': 'PCLYVM'
    }

    # Assignació de variables als datasets corresponents
    # for i in range(len(df_variables)):  # Assegurem-nos que l'índex comença a 0
    for i in range(len(df_variables)):
        category = df_variables.loc[i, 'Camp']
        if category in category_map:
            dataset_name = category_map[category]
            datasets[dataset_name][dict[i+1]] = df[dict[i+1]]

    return datasets


def split_datasets(datasets):
    train_test_splits = {}

    for name, df in datasets.items():
        if 'temps_fins_reincidencia1a' in df.columns:
            X = df.drop(columns='temps_fins_reincidencia1a')
            y = df['temps_fins_reincidencia1a']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=35)

            train_test_splits[name] = {
                'X_train': X_train, 'X_test': X_test, 
                'y_train': y_train, 'y_test': y_test
            }

    return train_test_splits


def split_datasets_clas(datasets):
    train_test_splits = {}

    for name, df in datasets.items():
        if 'Violent_o_lesions' in df.columns:
            X = df.drop(columns='Violent_o_lesions')
            y = df['Violent_o_lesions']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=35, stratify=y)

            train_test_splits[name] = {
                'X_train': X_train, 'X_test': X_test, 
                'y_train': y_train, 'y_test': y_test
            }

    return train_test_splits

def apply_pipeline_to_train_sets(train_test_splits):
    # Definir la pipeline de preprocessament
    preprocess_pipeline = Pipeline([
        ('neteja_na', NetejaNAColumns(llindar=0.6)),
        ('omple_nans', OmpleNans()),
        ('label_encoding', LabelEncoding())
    ])
    
    transformed_train_sets = {}

    # Aplicar la pipeline a cada dataset de training
    for name, data in train_test_splits.items():
        X_train_transformed = preprocess_pipeline.fit_transform(data['X_train'])
        X_test_transformed = preprocess_pipeline.transform(data['X_test'])

        transformed_train_sets[name] = {
            'X_train': X_train_transformed,
            'X_test': X_test_transformed,   
            'y_train': data['y_train'],
            'y_test': data['y_test']
        }

    return transformed_train_sets


def apply_significant_features_filter(train_test_splits, df, llindar=0.01, alpha=0.1, print_results=True):
    filtered_datasets = {}

    print("Entrem a la funció:")

    for name, data in train_test_splits.items():
        print(f'Applying significant features filter to dataset {name}...')

        print("Tamanys train",data['X_train'].shape)
        print("Tamanys train y",data['y_train'].shape)
        print("Tamanys train test",data['X_test'].shape)

        X_train = data['X_train']
        X_test = data['X_test']
        y_train = data['y_train']
        y_test = data['y_test']

        ## mirem el tamany de train
        print("Tamanys train",X_train.shape)
        print("Tamanys train y",y_train.shape)
        print("Tamanys train test",X_test.shape)


        # Aplicar la selecció de característiques significatives a X_train
        X_train_filtrat, coef_significatius = filter_significant_features(
            X_train, y_train, df, llindar=llindar, alpha=alpha, print_results=print_results
        )

        # Aplicar la mateixa selecció a X_test (mantenir només les columnes significatives)
        X_test_filtrat = X_test[coef_significatius.index]

        # Guardar resultats
        filtered_datasets[name] = {
            'X_train_filtrat': X_train_filtrat,
            'X_test_filtrat': X_test_filtrat,
            'coef_significatius': coef_significatius,
            'y_train': y_train,
            'y_test': y_test
        }

    return filtered_datasets

def apply_significant_features_filter_false(train_test_splits, df, llindar=0.01, alpha=0.1, print_results=False):
    filtered_datasets = {}

    print("Entrem a la funció:")

    for name, data in train_test_splits.items():
        print(f'Applying significant features filter to dataset {name}...')

        print("Tamanys train",data['X_train'].shape)
        print("Tamanys train y",data['y_train'].shape)
        print("Tamanys train test",data['X_test'].shape)

        X_train = data['X_train']
        X_test = data['X_test']
        y_train = data['y_train']
        y_test = data['y_test']

        ## mirem el tamany de train
        print("Tamanys train",X_train.shape)
        print("Tamanys train y",y_train.shape)
        print("Tamanys train test",X_test.shape)

        X_train_filtrat = pd.DataFrame(X_train, columns=X_train.columns)
        X_test_filtrat = pd.DataFrame(X_test, columns=X_test.columns)

        # Guardar resultats
        filtered_datasets[name] = {
            'X_train_filtrat': X_train_filtrat,
            'X_test_filtrat': X_test_filtrat,
            'coef_significatius': None,
            'y_train': y_train,
            'y_test': y_test
        }

    return filtered_datasets

## en aquest fitxer crearem diferetns dataset
import numpy as np

from funcions_net import *
import pandas as pd

def drop_all_columns(df, meta, df_variables, dict):

    # Recopilem totes les columnes a eliminar
    columns_to_drop = []
    columns_to_drop.extend(find_drop_columns(meta, df_variables, dict))
    columns_to_drop.extend(eliminem_preguntes(meta, df_variables, dict))
    columns_to_drop.extend(eliminem_items(meta, df_variables, dict))
    #columns_to_drop.extend(eliminem_mitjanes(meta, df_variables, dict))
    columns_to_drop.extend(eliminem_prob(meta, df_variables, dict))
    columns_to_drop.extend(eliminem_mitjanes2(meta, df_variables, dict))

    # Eliminem columnes sense errors
    df = df.drop(columns=columns_to_drop, errors='ignore')
    
    return df

def create_dataset(df, dict):
    """
    Crea diferents datasets a partir del dataset original:
      - df_net: DataFrame sense les columnes eliminades.
      - df_net_omplert: DataFrame sense les columnes eliminades i sense NaNs.
    
    Paràmetres:
        - df: DataFrame original.
        - meta: Metadades dels datasets.
        - df_variables: DataFrame amb les variables i els seus valors.
        - dict: Diccionari amb els noms de les columnes i els seus valors.
    
    Retorna:
        - df_net: DataFrame sense les columnes eliminades.
        - df_net_omplert: DataFrame sense les columnes eliminades i sense NaNs.
    """
    # Eliminem totes les columnes que no ens interessen
    num_files_afegir = [6,15,16,17,18,19,66,179,184,210,235,260,285,305,306,307,512,937,1057,1151,1152,1153]
    for i in range(515, 527):
        num_files_afegir.append(i)
    for i in range(886, 907):
        num_files_afegir.append(i)
    for i in range(1179, 1185):
        num_files_afegir.append(i)
    num_files2 = [1265,1484,1481,1485,1486,1555]
    num_5 = [71,72]

    num_files_total = np.concatenate((num_files_afegir,num_files2))

    ## trobem el nom de les columnes
    nom_col = []

    for num in num_files_total:
        nom_col.append(dict[num])
    df_final = df[nom_col]


    for num in num_5:
        df_final[dict[num]] = np.where(df[dict[num]] == 5, 1, 0)
    
    return df_final

def columnes_més_importants(df,num_agafar=40):
    variables = [
    "temps_fins_reincidencia1a",
    "Reincidencia_Juvenil_preCAPE",
    "EdatAD",
    "SAVRYSoc",
    "EdatD1mesuraInternament",
    "Pri_Ing",
    "EdatT1mesuraInternament",
    "Nom_Exp",
    "Total_Del",
    "PCLxFAC_3",
    "PCLjFAC_3",
    "t_internamentCape",
    "SAVRYTot",
    "nombre_fets_previs",
    "nDelictesViolents",
    "Del_Condem",
    "fets_condemnat_preCAPE",
    "CAPEe_PT",
    "SAVRYInd",
    "intensitat_delictiva_total",
    "TRFMSocProbCat",
    "TRFT_AP",
    "YSRExtProbT",
    "PCLoFAC_3",
    "Pri_Exp"
    ]
    variables_2 = [
    "YSRTotProbT",
    "YSRRulBeh",
    "VASTDir_3gr",
    "TempConeixTutor",
    "YSRExtProb",
    "TempConeixMestre",
    "Edat1erDelicte",
    "YSRDSMODD",
    "YSRDSMCD",
    "PCLeFAC_3",
    "YSRDSMCDT",
    "TRFMWithDepCat",
    "YSRTotProbCat",
    "TRFTExtProbCat",
    "YSRTotProb"
]
    
    variables.extend(variables_2)
    df_final = df[variables[:num_agafar]]
    return df_final

def create_dataset_2(df, dict):
    # Eliminem totes les columnes que no ens interessen
    num_files_afegir = [6,15,16,17,18,19,66,179,184,210,235,260,285,305,306,307,512,937,1057,1151,1152,1153]
    for i in range(515, 527):
        num_files_afegir.append(i)
    for i in range(886, 907):
        num_files_afegir.append(i)
    for i in range(1179, 1185):
        num_files_afegir.append(i)
    num_files2 = [1265,1484,1481,1485,1486,1555]
    num_5 = [71,72]
    num_delictes = list(range(24, 65))

    num_files_total = np.concatenate((num_files_afegir,num_files2))

    ## trobem el nom de les columnes
    nom_col = []

    for num in num_files_total:
        nom_col.append(dict[num])
    df_final = df[nom_col]

    print("Hola")

    df_final['Delictes molt violents'] = 0
    for num in num_5:
        df_final[dict[num]] = np.where(df[dict[num]] == 5, 1, 0)
    ##for num in num_delictes:
        ##df_final['Delictes_molt_violents'] += np.where((df[dict[num]] > 3) & (df[dict[num]] < 11), 1, 0)

    df_final['Delictes_molt_violents'] = np.where((df[dict[num]] > 3) & (df[dict[num]] < 11), 1, 0)

    return df_final

def create_dataset_global(df_orig, dict_vars,df_variables,meta):
    df_psico = dataset_psicologia(df_orig, dict_vars, df_variables)
    df = drop_all_columns(df_psico, meta, df_variables, dict_vars)

    ## dataset inicial
    df_inicial = dataset_inicial(df_orig, dict_vars)

    num_files_afegir = [16, 17, 18, 1484, 1485, 1486, 1495, 1496, 1497, 1583]

    # Obté els noms de les columnes corresponents a les claus
    nom_col = [dict_vars[num] for num in num_files_afegir if num in dict_vars]

    # Filtra les columnes amb valors escalars
    col_segures = [col for col in nom_col if pd.api.types.is_scalar(df_orig[col].dropna().iloc[0])]

    # Crea el DataFrame només amb les columnes segures
    df_extra = df_orig[col_segures]


    print("Files df:", df.shape[1])
    print("Files df_extra:", df_extra.shape[1])
    print("Files df_inicial:", df_inicial.shape[1])

    df_extra = df_extra[[col for col in df_extra.columns if col not in df.columns]]

    # Elimina de df_inicial les columnes que ja són a df (i a df_extra també, per si de cas)
    df_inicial = df_inicial[[col for col in df_inicial.columns if col not in df.columns and col not in df_extra.columns]]

    df = df.reset_index(drop=True)
    df_inicial = df_inicial.reset_index(drop=True)
    df_extra = df_extra.reset_index(drop=True)

    df_global = pd.concat([df, df_extra, df_inicial], axis=1)


    duplicades = df_global.columns[df_global.columns.duplicated()]
    if not duplicades.empty:
        print("⚠️ Columnes duplicades després del concat:", list(duplicades))

    return df_global



def origen(df_actual,df_anterior):
    df_actual['Gitanos'] = np.where(df_anterior['Etnia'] ==1, 1, 0)
    df_actual['Magrebins'] = np.where(df_anterior['Etnia'] ==2, 1, 0)
    df_actual['Subsaharians'] = np.where(df_anterior['Etnia'] ==3, 1, 0)
    df_actual['Llatins'] = np.where(df_anterior['Etnia'] ==4, 1, 0)
    df_actual['Caucasics'] = np.where(df_anterior['Etnia'] ==5, 1, 0)
    df_actual['Asiatics'] = np.where(df_anterior['Etnia'] ==6, 1, 0)
    df_actual['Eslaus'] = np.where(df_anterior['Etnia'] ==7, 1, 0)

    df_actual['Espanya'] = np.where(df_anterior['nacionalitat_agrupat'] == 1, 1, 0)
    df_actual['Unio_Europea'] = np.where(df_anterior['nacionalitat_agrupat'] == 2, 1, 0)
    df_actual['Resta_Europa'] = np.where(df_anterior['nacionalitat_agrupat'] == 3, 1, 0)
    df_actual['Magreb'] = np.where(df_anterior['nacionalitat_agrupat'] == 4, 1, 0)
    df_actual['Resta_Africa'] = np.where(df_anterior['nacionalitat_agrupat'] == 5, 1, 0)
    df_actual['Centre_Sud_America'] = np.where(df_anterior['nacionalitat_agrupat'] == 6, 1, 0)
    df_actual['Asia'] = np.where(df_anterior['nacionalitat_agrupat'] == 7, 1, 0)
    df_actual['Resta_Mon'] = np.where(df_anterior['nacionalitat_agrupat'] == 8, 1, 0)
    df_actual['Nord_America'] = np.where(df_anterior['nacionalitat_agrupat'] == 9, 1, 0)

    df_actual['Pares_Catalans'] = np.where(df_anterior['OrgnFam'] == 1, 1, 0)
    df_actual['Pare_Catala_Altres_Espanya'] = np.where(df_anterior['OrgnFam'] == 2, 1, 0)
    df_actual['Pare_Catala_Fora_Espanya'] = np.where(df_anterior['OrgnFam'] == 3, 1, 0)
    df_actual['Pares_No_Catalans_Espanya'] = np.where(df_anterior['OrgnFam'] == 4, 1, 0)
    df_actual['Pares_No_Catalans_Un_Espanya'] = np.where(df_anterior['OrgnFam'] == 5, 1, 0)
    df_actual['Pares_Fora_Espanya'] = np.where(df_anterior['OrgnFam'] == 6, 1, 0)

    return df_actual

def dataset_psicologia(df,dict,df_variables):
    # Get psychological assessment columns
    psych_columns = []
    

    psych_test_nums = list(range(1, 1590))
    for i in range(1590):
        if df_variables['Camp'][i] == 'FET':
            psych_test_nums.remove(i+1)
        if df_variables['Camp'][i] == 'FETS':
            psych_test_nums.remove(i+1)
        if df_variables['Camp'][i] == 'Generic':
            psych_test_nums.remove(i+1)
        elif df_variables['Camp'][i] == 'GEN':
            psych_test_nums.remove(i+1)
        elif df_variables['Camp'][i] == 'Delictes':
            psych_test_nums.remove(i+1)
        elif df_variables['Camp'][i] == 'Delitctes':
            psych_test_nums.remove(i+1)
        elif df_variables['Camp'][i] == 'INT':
            psych_test_nums.remove(i+1)
        elif df_variables['Camp'][i] == 'Generic':
            psych_test_nums.remove(i+1)
        elif df_variables['Camp'][i] == 'Sanc':
            psych_test_nums.remove(i+1)
        elif df_variables['Camp'][i] == 'Fets':
            psych_test_nums.remove(i+1)
        elif df_variables['Camp'][i] == 'REIN':
            psych_test_nums.remove(i+1)
        elif df_variables['Camp'][i] == 'INGRES':
            psych_test_nums.remove(i+1)
        elif df_variables['Camp'][i] == 'Sancions':
            psych_test_nums.remove(i+1)
        elif df_variables['Camp'][i] == 'SANC':
            psych_test_nums.remove(i+1)
        elif df_variables['Camp'][i] == 'CAP':
            psych_test_nums.remove(i+1)
        elif df_variables['Camp'][i] == 'Entrevista':
            psych_test_nums.remove(i+1)
        elif df_variables['Camp'][i] == 'NaN':
            psych_test_nums.remove(i+1)
        elif df_variables['Camp'][i] == 'SAVRY':
            psych_test_nums.remove(i+1)

    for num in psych_test_nums:
        psych_columns.append(dict[num])
    
    # Create the psychological assessment dataset
    df_psych = df[psych_columns]
    
    return df_psych


def delictes_comesos(df,dict):
    # Inicialitzar les columnes comptadores per evitar errors si no existeixen
    categories = {"Robatori":1,"Robatori amb violència": 2,"Drogues": 3,"Lesions": 4,"Homicidi": 5,
    "Delictes sexuals": 6,"Seguretat vial": 7,"Delictes patrimonials": 8,"Quebrantament de condemna": 9,
    "Segrest": 10,"Incendi": 11,"Administració de justícia": 12,"Seguretat de l'Estat": 13,"Miscel·lània": 14
    }

    
    df_delictes = pd.DataFrame(0, index=df.index, columns=categories.keys())

    # Cal crear les columnes
    for categoria in categories.keys():
        df_delictes[categoria] = 0

    # Recórrer les columnes delictives i sumar a la categoria corresponent
    for i in range(24, 65, 2):
        for categoria, valor in categories.items():
            df_delictes[categoria] += (df[dict[i]] == valor).astype(int)

    # df_delictes["Delictes molt violents"] = df_delictes["Homicidi"] + df_delictes["Lesions"] + df_delictes["Delictes sexuals"] + df_delictes["Segrest"]
    df_delictes["Violent_o_lesions"] = (
    (df_delictes["Homicidi"] > 0) |
    (df_delictes["Delictes sexuals"] > 0) |
    (df_delictes["Segrest"] > 0) |
    (df_delictes["Lesions"] > 2)
)



    print("Delictes comesos")
    print(df_delictes.shape)

    return df_delictes

        


def dataset_inicial(df,dict):
    ## Volem aconseguir les variables que tenen a veure abans del primer internament
    ## dades generiques
    print("Creant dataset inicial")
    index = [2,6,15,19,66,67,68,69,70,75,76,77,78,79,81]
    df_inicial = df[[dict[i] for i in index]]

    print("Creant dataset origen")
    print(df_inicial.shape)

    
    df_inicial = origen(df_inicial,df)

    print("Creant dataset origen")
    print(df_inicial.shape)
    

    df_delcites = delictes_comesos(df,dict)

    print("Dataset final")
    print(df_delcites.shape)

    ## concatenem els dos datasets
    df_inicial = pd.concat([df_inicial,df_delcites],axis=1)
    
    print("Dataset final")
    print(df_inicial.shape)
    
    return df_inicial

def dataset_diferent_2(df, dict, df_variables):
    # Inicialització dels datasets
    def initialize_datasets(df):
        datasets = {}
        dataset_names = [
            'IQ', 'SAVRY', 'VAST', 'PCL', 'CAPE', 'YPI', 'RPQ', 'CCA', 'SD3', 
            'ICUJ', 'ICUT', 'TRFM', 'YSR', 'TriPM', 'TRFT'
        ]
        
        for name in dataset_names:
            datasets[name] = pd.DataFrame(index=df.index)
            datasets[name]['temps_fins_reincidencia1a'] = df['temps_fins_reincidencia1a'].apply(lambda x: 1095 if x > 1095 else x)
        
        return datasets

    datasets = initialize_datasets(df)


    # Mapeig de categories a datasets
    category_map = {
        'Test IQ': 'IQ', 'SAVRY': 'SAVRY', 'VAST': 'VAST', 'YPI': 'YPI',
        'RPQ': 'RPQ', 'CCA': 'CCA', 'SD3': 'SD3', 'ICUJ': 'ICUJ',
        'ICUT': 'ICUT', 'TRFM': 'TRFM', 'YSR': 'YSR', 'TriPM': 'TriPM',
        'TRFT': 'TRFT', 'TRFT DMS': 'TRFT', 'CAPe': 'CAPE', 'CAPo': 'CAPE',
        'PCLe': 'PCL', 'PCLo': 'PCL', 'PCLj': 'PCL', 'PCLx': 'PCL', 'PCLYVM': 'PCL'
    }


    # Assignació de variables als datasets corresponents
    for i in range(len(df_variables)):  # Assegurem-nos que l'índex comença a 0
        category = df_variables.loc[i, 'Camp']
        if category in category_map:
            dataset_name = category_map[category]
            datasets[dataset_name][dict[i+1]] = df[dict[i+1]]

    return datasets




    

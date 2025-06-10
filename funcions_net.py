

## Aquesta funció a partir de un csv i el dataframe i crea un diccionari
## aquest diccionari té com a clau el numero de la columna i com a valor el nom de la variable
def create_dict(meta):
    dict = {}
    for i in range(1, len(meta.column_names)+1):
        dict[i] = meta.column_names[i-1]
    return dict

def find_drop_columns(meta, df_variables,dict):
    variables_to_drop = []
    for i in range(1, len(dict)+1):
        if df_variables["Utilitat"][i-1] != 'SI':
            variables_to_drop.append(dict[i])
    variables_to_drop.remove('temps_fins_reincidencia1a')
    return variables_to_drop

def drop_columns(df, columns_to_drop):
    df = df.drop(columns=columns_to_drop,errors='ignore')
    return df

def eliminem_preguntes(meta, df_variables,dict):
    variables_to_drop = []
    for i in range(1, len(dict)+1):
        if df_variables["Que és"][i-1] == 'PREG':
            variables_to_drop.append(dict[i])
    return variables_to_drop

def eliminem_items(meta, df_variables,dict):
    variables_to_drop = []
    for i in range(1, len(dict)+1):
        if df_variables["Que és"][i-1] == 'Item':
            variables_to_drop.append(dict[i])
    return variables_to_drop

def eliminem_mitjanes(meta, df_variables,dict):
    variables_to_drop = []
    for i in range(1, len(dict)+1):
        if df_variables["Que és"][i-1] == 'Mitjanes' :
            variables_to_drop.append(dict[i])
    return variables_to_drop

def eliminem_prob(meta, df_variables,dict):
    variables_to_drop = []
    for i in range(1, len(dict)+1):
        if df_variables["Que és"][i-1] == 'Prob':
            variables_to_drop.append(dict[i])
    return variables_to_drop

def eliminem_mitjanes2(meta, df_variables,dict):
    variables_to_drop = []
    for i in range(1, len(dict)+1):
        if df_variables["Que és"][i-1] == 'Mitjanes(agr)' :
            variables_to_drop.append(dict[i])
    return variables_to_drop



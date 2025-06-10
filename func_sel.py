import pandas as pd
from sklearn.linear_model import Lasso

def filter_significant_features(X_train,y_train, df, llindar=0.1,alpha=1, print_results=True):
    lasso = Lasso(alpha)
    lasso.fit(X_train, y_train)

    ## treim els coeficients que són 0

    coef = pd.Series(lasso.coef_, index = X_train.columns)
    if print_results:
        with open("resultats_lasso.txt", "a") as f:
            print("HOlaaaaa")
            f.write("Hem fet el model Lasso\n")
            f.write(f"Hem trobat {sum(coef == 0)} coeficients nuls\n")
            f.write(f"Hem trobat {sum(coef != 0)} coeficients no nuls\n")

            coef_no_nuls = coef[coef != 0]
            coef_ordenats = coef_no_nuls.reindex(coef_no_nuls.abs().sort_values(ascending=False).index)

            f.write("Els coeficients no nuls ordenats per valor absolut són:\n")
            f.write(str(coef_ordenats[0:25]) + "\n")
            
    coef_significatius = coef[abs(coef) > llindar]
    
    # Eliminar les columnes amb coeficients baixos o nuls del DataFrame original
    X_train = pd.DataFrame(X_train, columns=X_train.columns)
    X_train_filtrat = X_train[coef_significatius.index]

    return X_train_filtrat, coef_significatius


def filter_significant_features_2(X_train,y_train, df, llindar=0.1,alpha=1, print_results=True):
    lasso = Lasso(alpha)
    lasso.fit(X_train, y_train)

    ## treim els coeficients que són 0

    coef = pd.Series(lasso.coef_, index = X_train.columns)
    if print_results:
        print("Hem fet el model Lasso")
        print("Hem trobat ", sum(coef == 0), " coeficients nuls")
        print("Hem trobat ", sum(coef != 0), " coeficients no nuls")
        ## fem print dels que no són 0
        print("Els coeficients no nuls són:")
        print(coef[coef != 0])
        ## fem print dels coeficients nuls
        print("Els coeficients nuls són:")
        print(coef[coef == 0])
        # Assuming 'coef' is defined elsewhere and available
        # Filtrar els coeficients que són significatius (absolut > llindar)
    coef_significatius = coef[abs(coef) > llindar]
    
    # Eliminar les columnes amb coeficients baixos o nuls del DataFrame original
    X_train = pd.DataFrame(X_train, columns=X_train.columns)
    X_train_filtrat = X_train[coef_significatius.index]

    return X_train_filtrat, coef_significatius
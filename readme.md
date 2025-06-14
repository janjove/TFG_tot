# Predicció de la reincidència criminal amb *Machine Learning*

Aquest repositori és el codi del meu tfg, inicialment ho vaig fer tot amb notebooks, a la llarga totes elles molt caòtiques d'on s'han extret els resultats finals es els fitxers predictor_*.py, que implementen un pipeline complet.
Sé que el repositori pot semblar caòtic al principi però per estalvbiar temps i fer els codi el mes entenedor per mi ho he estructurat aixi potser en un futur ho canvio per executar en un sol fitxer reduir les funcions, de moment disculpeu les molèsties.

## Metodologia general

La metodologia general del projecte està explicada en el treball de fi de grau, però aquí es fa un resum de les etapes principals:
# Metodologia del TFG

Aquest document resumeix els passos seguits en la metodologia del Treball de Fi de Grau, centrada en la construcció d’un model predictiu amb anàlisi d’errors i estratègies de millora.

## 1. Procés principal (bucle general d’entrenament)
S’inicia amb les dades i es construeix un model predictiu seguint un cicle d’entrenament-estimació-validació.  
_(Aquest és el camí central que connecta tots els passos del modelatge)._

## 2. Prova de diferents models predictius
Durant el procés es comparen diversos enfocaments:
- Model general
- Mètode basat en psicologia
- Mètode inicial  
_(Aquesta és una bifurcació que explora diferents arquitectures o estratègies de modelatge)._

## 3. Selecció automàtica de variables
Per optimitzar l’entrada de dades, s’apliquen tècniques com:
- Lasso
- RFE
- ENNS  
_(Aquesta branca selecciona quines variables són més rellevants per millorar el model)._

## 4. Augmentació o generació de dades
Per millorar la qualitat i robustesa del model, es fan servir estratègies com:
- Bootstrapping
- TGAN (generació sintètica amb xarxes)
- Combinació de mètodes  
_(Aquest pas incrementa o millora les dades d’entrenament)._

## 5. Anàlisi i interpretació de l’error
Després d’entrenar els models, s’analitzen i validen els resultats en dues branques:

### Anàlisi qualitativa
- Interpretació de coeficients
- Explicació de casos concrets
- Ús de LIME per fer el model més comprensible

### Comparació quantitativa d’errors
Validació creuada:
- Mateixes dades amb mètodes diferents
- Dades diferents amb el mateix mètode

## Estructura de carpetes i fitxers

| Fitxer / Carpeta                | Descripció                                                                                                          |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **notebooks/**                  | Proves inicials, exploracions i prototips dels models.                                                              |
| **crear\_dataset.py**           | Genera els diferents datasets de base a partir de dades brutes.                                                     |
| **dataset\_topy.py**            | Variant que llança la seqüència completa (semblant a `predictor.py`) seprarant per test   |
| **datasets\_separats.py**       | Funcions utilitzades per tractar els models amb datsets separats                                    |
| **enns.py**                     | Implementació d’**ENNS** per reequilibrar classes.                                       |
| **enns2.py**                    | Mateixa funcionalitat que `enns.py` però amb modificacions menors i millores.                                       |
| **explicatiu\_test.py**         | Proves per validar el model, i explaicar les variables                                              |
| **func\_sel.py**                | Funcions de **selecció de variables** (LASSO, RFE…) i descart de variables irrellevants definides en un full Excel. |
| **funcions\_net.py**            | Utilitats genèriques utilitzades arreu del projecte (logging, mètriques, helpers de xarxa, etc.).                   |
| **lime\_test.py**               | Proves unitàries sobre l’aplicació de LIME.                                                                         |
| **model\_explicatiu.py**        | Model que retorna a una carpteta, els coeficients de cada variable                  |
| **predictor.py**                | Script principal: executa la etodologia bàsica        |
| **predictor\_RFE.py**           | Variant amb **Recursive Feature Elimination**.                                                                      |
| **predictor\_TGAN.py**          | Variant que genera dades sintètiques amb **TGAN**.                                                            |
| **predictor\_bootstrap.py**     | Variant que aplica **bootstrap** a l’entrenament.                                                          |
| **predictor\_bootstrapTGAN.py** | Combina *bootstrap* + *TGAN*.                                                                                       |
| **predictor\_enns.py**          | Variant que utilitza dataset reequilibrat amb **ENNS**.                                                              |
| **preprocessing.py**            | Conjunt de funcions generals de preprocessament (neteja, imputació, *scaling*…).                                    |
| **sel\_variables.py**           | Implementa diversos mètodes de selecció de variables (LASSO, RFE, enns).                       |

## Execució del codi
El codi no es pot executar, ja que està preparat especificament per les dades del tfg, aquestes dades no poden ser públiques. 
Tot i tenir les dades, el codi contindrà errors, ja que estan preparades per un analisis de les variables extres que vaig fer.
Està explicat en el tfg, però per resumir, vaig etiquetar les variables per útils, esbiaxades, futures o irrellevants.
també estan classificades per el tipus de test que son o el tipus si son una pregunta, una mitjana, un item o un puntuació total.


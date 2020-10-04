# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 15:31:40 2020

@author: marcelo
"""

import pandas as pd
import pyodbc
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import roc_auc_score

# cnxn = pyodbc.connect("Driver={SQL Server Native Client 11.0};"
#                       "Server=localhost\\NOVAINSTANCIA2;"
#                       "Database=Ind_Economicos;"
#                       "Trusted_Connection=yes;")

# query = "SELECT * From [Fundos_TCC].[Fundos_05052020].[fundos_dados]"
# dfFundos = pd.read_sql(query, cnxn)
# dfFundos.to_csv('fundos_dados.csv', index = False)

# Carregando os daods dos fundos
dfFundos = pd.read_csv('fundos_dados.csv')

dfFundos.head()
# 36147 fundos de investimento sem repetições
dfFundos.shape
len(np.unique(dfFundos['cnpj']))

dfFundos.dtypes
dados_vaziosXColuna = dfFundos.isna().sum(axis = 0)
dados_vaziosXColuna

# Vamos apagar as colunas com mais de 35% dos dados faltantes
colApagar = np.array(dfFundos.columns[(dados_vaziosXColuna > (len(dfFundos) * 0.35))])
dfFundos.shape
dfFundos.drop(colApagar, axis=1, inplace=True)
dfFundos.shape
dados_vaziosXColuna = dfFundos.isna().sum(axis = 0)
dados_vaziosXColuna

# Preenchendo os campos nulos
dfFundos['cnpjAdmin'].fillna('N/A', inplace=True)
dfFundos[dfFundos['cond_aberto'].isna()]
dfFundos = dfFundos.drop(dfFundos[dfFundos['cond_aberto'].isna()].index)
dados_vaziosXColuna = dfFundos.isna().sum(axis = 0)
dados_vaziosXColuna

# A tabela com os índices de rentabilidade possui 40830775 registros e se mostrou inviável trazer o registros
# para a memória da aplicação.
# query = """Select rentInicial.cnpj, rentInicial.DataInicial, rentFinal.DataFinal, diasOperacao, ((indFinal/indInicial)-1)*100 as rentabilidade, ((rentFinal.indCDI/rentInicial.indCDI)-1)*100 as rentabilidadeCDI From 
#   (Select indDiarios.cnpj, data as DataInicial, rentabilidade as indInicial, indCDI, diasOperacao From [Fundos_TCC].[Fundos_05052020].[indicesDiarios] indDiarios
#   INNER JOIN (Select cnpj, Min(data) as DataInicial, count(*) as diasOperacao From [Fundos_TCC].[Fundos_05052020].[indicesDiarios] Group By cnpj) AS inicioFundo
#  	ON indDiarios.cnpj = inicioFundo.cnpj AND indDiarios.data = inicioFundo.DataInicial
#   INNER JOIN (Select Ind_Data as DataCDI, Ind_Rent as indCDI From [Fundos_TCC].[dbo].[indicesDiariosCDI]) AS inicioCDI
#  	ON indDiarios.data = inicioCDI.DataCDI) rentInicial
#  	INNER JOIN 
#  	  (Select indDiarios.cnpj, data as DataFinal, rentabilidade as indFinal, indCDI From [Fundos_TCC].[Fundos_05052020].[indicesDiarios] indDiarios
#  	INNER JOIN (Select cnpj, Max(data) as DataFinal From [Fundos_TCC].[Fundos_05052020].[indicesDiarios] Group By cnpj) AS finalFundo
# 		ON indDiarios.cnpj = finalFundo.cnpj AND indDiarios.data = finalFundo.DataFinal
#  	INNER JOIN (Select Ind_Data as DataCDI, Ind_Rent as indCDI From [Fundos_TCC].[dbo].[indicesDiariosCDI]) AS finalCDI
# 		ON indDiarios.data = finalCDI.DataCDI) rentFinal
#  	ON rentInicial.cnpj = rentFinal.cnpj"""
# dfIndDiarios = pd.read_sql(query, cnxn)
# dfIndDiarios.to_csv('indices_diarios.csv', index = False)

# Carregando as rentabilidades dos fundos e do CDI durante o período de vigência do fundo
dfIndDiarios = pd.read_csv('indices_diarios.csv')
dfIndDiarios.shape
len(np.unique(dfIndDiarios['cnpj']))

# Este join excluiu 2.496 fundos de investimento
dfFundosRent = pd.merge(dfFundos, dfIndDiarios, on='cnpj')
dfFundosRent.shape
len(np.unique(dfFundosRent['cnpj']))

# Criando o campo AcimaCDI e AcimaCDILabel que indicma se o fundo conseguiu rentabilidade acima do CDI 
# durante o seu tempo de operação.
dfFundosRent['AcimaCDI'] = dfFundosRent['rentabilidade'] > dfFundosRent['rentabilidadeCDI']
dfFundosRent['AcimaCDILabel'] = dfFundosRent['AcimaCDI'].apply(lambda acimaCDI: "Acima do CDI" if acimaCDI==True else "Abaixo do CDI")

# Calculando o percentual de fundos acima e abaixo do CDI
percentAcimaCDI = dfFundosRent['AcimaCDI'].sum()/len(dfFundosRent['AcimaCDI'])
percentAbaixoCDI = (~dfFundosRent['AcimaCDI']).sum()/len(dfFundosRent['AcimaCDI'])

# Exibindo o gráfico mostrando a quantidade fundos acima do CDI
dfFundosRent.AcimaCDILabel.value_counts().plot(kind='bar', title ="Desempenho dos Fundos")

# Exibindo um boxplot e um histograma com o tempo em que os fundos permanecem operacionais
# Cálculo da média de anos em que os fundos permanecem operacionais
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.boxplot(dfFundosRent['diasOperacao']/251)
ax2.set_title("Tempo de operação do fundo")
ax2.set_xlabel('Anos de operação')
ax2.set_ylabel('Qtde')
ax2.hist(dfFundosRent['diasOperacao']/251)

# Cálculo da média de anos em que os fundos permanecem operacionais
mediaAnos = dfFundosRent['diasOperacao'].mean()/251
mediaAnos

# Criando o Dataframe com fundos que possuem pelo menso 4 anos de operação
dfFundosRent4Anos = dfFundosRent[dfFundosRent['diasOperacao'] > 251*4]
dfFundosRent4Anos.shape

percentAcimaCDI = dfFundosRent4Anos['AcimaCDI'].sum()/len(dfFundosRent4Anos['AcimaCDI'])
percentAbaixoCDI = (~dfFundosRent4Anos['AcimaCDI']).sum()/len(dfFundosRent4Anos['AcimaCDI'])

# Exibindo o gráfico mostrando a quantidade fundos acima do CDI
dfFundosRent4Anos.AcimaCDILabel.value_counts().plot(kind='bar', title ="Desempenho dos Fundos com pelo menos 4 anos de operação")
# Agrupando e exibindo os fundos por tipo 
dfFundosRent4Anos.groupby('classe_p_n').AcimaCDILabel.value_counts().unstack().plot(kind='bar', title ="Desempenho dos fundos por tipo")
# Agrupando e exibindo os fundos exclusivos
dfFundosRent4Anos.groupby('exclusivo').AcimaCDILabel.value_counts().unstack().plot(kind='bar', title ="Desempenho dos fundos exclusivos")
# Agrupando e exibindo os fundos restritos
dfFundosRent4Anos.groupby('restrito').AcimaCDILabel.value_counts().unstack().plot(kind='bar', title ="Desempenho dos fundos restritos")
# Agrupando e exibindo os fundos para investidores qualificados
dfFundosRent4Anos.groupby('invest_qualif').AcimaCDILabel.value_counts().unstack().plot(kind='bar', title ="Desempenho dos fundos para investidores qualificados")

# Carga dos gestores
# query = "SELECT cnpjFundo, cnpjGestor, nome as nomeGestor From [Fundos_TCC].[Fundos_05052020].[fundosXgestores] fg INNER JOIN [Fundos_TCC].[Fundos_05052020].[gestores] gest ON fg.cnpjGestor = gest.cnpj"
# dfListaGestores = pd.read_sql(query, cnxn)
# dfListaGestores.to_csv('fundosXgestores.csv', index = False)
dfListaGestores = pd.read_csv('fundosXgestores.csv')

dfFundosGestores = pd.merge(dfFundosRent4Anos, dfListaGestores, how='left',  left_on='cnpj', right_on='cnpjFundo')
dfFundosGestores.shape


# Análise dos Gestores
# Gestores que gerenciam pelo menos 50 fundos
dfGestMais50Fundos = dfFundosGestores[dfFundosGestores['cnpjGestor'].isin(dfFundosGestores['cnpjGestor'].value_counts()[dfFundosGestores['cnpjGestor'].value_counts()>50].index)]
dfGestMais50Fundos.shape
dfGestMais50Fundos['cnpjGestor'].unique()
dfGestMais50Fundos['nomeGestorAbrev'] = dfGestMais50Fundos['nomeGestor'].str[:40]
percentAcimaCDI = dfGestMais50Fundos['AcimaCDI'].sum()/len(dfGestMais50Fundos['AcimaCDI'])
percentAbaixoCDI = (~dfGestMais50Fundos['AcimaCDI']).sum()/len(dfGestMais50Fundos['AcimaCDI'])
# Exibindo o gráfico mostrando a quantidade fundos acima do CDI
dfGestMais50Fundos.AcimaCDILabel.value_counts().plot(kind='bar', title ="Desempenho dos fundos dos maiores gestores")
# Agrupando os fundos por gestor 
dfGestMais50Fundos.groupby('nomeGestorAbrev').AcimaCDILabel.value_counts().unstack().plot(kind='bar', title ="Desempenho dos fundos por gestor")

# Filtrando os melhores gestores
dfGestDesemp = dfGestMais50Fundos.groupby(['nomeGestor', 'AcimaCDI']).size().unstack(fill_value=0)
dfGestDesemp = dfGestDesemp.reset_index()
dfBonsGestores = dfGestDesemp[dfGestDesemp[False]<dfGestDesemp[True]]

# Vamos apagar as colunas 'cnpj', 'nome', 'nome_abreviado', 'unique_slug', 'benchmark', 'indice' 
# 'DataInicial', 'DataFinal', 'diasOperacao', 'last_update', 'last_quote_date', 'first_quote_date', 
# 'situacao', 'status', 'rentabilidade', 'rentabilidadeCDI' 
colApagar = ['cnpj', 'cnpjFundo', 'nome', 'nome_abreviado', 'unique_slug', 'benchmark', 'indice', 'DataInicial', 
             'DataFinal', 'diasOperacao', 'last_update', 'last_quote_date', 'first_quote_date', 
             'situacao', 'status', 'rentabilidade', 'rentabilidadeCDI', 'patrimonio', 'cotistas', 'classe_p_s',
			 'AcimaCDILabel', 'nomeGestor' ]
dfFundosGestores.shape
dfFundosGestores.drop(colApagar, axis=1, inplace=True)
dados_vaziosXColuna = dfFundosGestores.isna().sum(axis = 0)
dados_vaziosXColuna
dfFundosGestores.shape
dfFundosGestores.head()

# Separando a variável dependente AcimaCDI das variáveis independentes.
resultadosFundo =dfFundosGestores['AcimaCDI'].astype('int')
dadosFundo = dfFundosGestores.drop(['AcimaCDI'], axis=1)
dadosFundo.dtypes

dados_vaziosXColuna = dadosFundo.isna().sum(axis = 0)
dados_vaziosXColuna

dadosFundo['cnpjGestor'].fillna('N/A', inplace=True)

dados_vaziosXColuna = dadosFundo.isna().sum(axis = 0)
dados_vaziosXColuna

# Alterando os tipos de alguns campos para que os algoritmos de ML não os considerem como varíaveis contínua e sim categóricas.
dadosFundo['de_cotas'] = dadosFundo['de_cotas'].astype('int')
dadosFundo['cond_aberto'] = dadosFundo['cond_aberto'].astype('int')
dadosFundo['exclusivo'] = dadosFundo['exclusivo'].astype('int')
dadosFundo['invest_qualif'] = dadosFundo['invest_qualif'].astype('int')
dadosFundo['tipo_de_previdencia'] = dadosFundo['tipo_de_previdencia'].astype('int')
dadosFundo['tribut_lp'] = dadosFundo['tribut_lp'].astype('int')
dadosFundo['restrito'] = dadosFundo['restrito'].astype('int')
dadosFundo['cnpjAdmin'] = dadosFundo['cnpjAdmin'].astype('str')
dadosFundo['cnpjGestor'] = dadosFundo['cnpjGestor'].astype('str')

# Transformando as variáveis categóricas em Dummies
dadosFundoDummies = pd.get_dummies(dadosFundo)
dados_vaziosXColuna = dadosFundoDummies.isna().sum(axis = 0)
dados_vaziosXColuna.sum()
colApagar = ['cnpjGestor_N/A', 'cnpjAdmin_N/A' ]
dadosFundoDummies.shape
dadosFundoDummies.drop(colApagar, axis=1, inplace=True)


# Função para retornar as importâncias de cada um dos atributos/campos para o algoritmo ExtraTreesClassifier.
def get_importancias(entrada, saida):
  modelo = ExtraTreesClassifier(n_estimators = 200, criterion = 'entropy', verbose = 0)
  modelo.fit(entrada, np.array(saida).ravel())
  
  importancias = modelo.feature_importances_
  atributos = np.array(entrada.columns)
  indices = np.argsort(importancias)[::-1]
  importancias = importancias[indices]
  atributos = atributos[indices]
  
  return atributos, importancias

atributos, importancias = get_importancias(dadosFundoDummies, resultadosFundo)

importancias

# Função que soma e mostra o gráfico de acumulação das importâncias.
def soma_grafico_importancias(atributos, importancias):
  soma_importancias = pd.DataFrame()
  for i in range(importancias.size):
    soma = importancias[:(i+1)].sum()
    aux = pd.Series([atributos[i], soma])
    soma_importancias = soma_importancias.append(aux, ignore_index = True)
  plt.scatter(soma_importancias.index, soma_importancias.iloc[:,1])
  plt.bar(atributos[:30], importancias[:30], color ='maroon',  width = 0.4)
  plt.xlabel("Atributos")
  plt.ylabel("Importãncias")
  plt.title("Importâncias dos 30 primeiros atributos") 
  plt.xticks(rotation='vertical')
  return soma_importancias

soma_importancias = soma_grafico_importancias(atributos, importancias)

soma_importancias

def atrib_manter(df, soma_importancias, threshold):
  lista_atrib_manter = list(soma_importancias[soma_importancias.iloc[:, 1] <= threshold].iloc[:, 0])
  dfManter = df.loc[:, lista_atrib_manter]
  
  return dfManter

dadosFundoDummies.shape

dadosFundoDummies = atrib_manter(dadosFundoDummies, soma_importancias, threshold = 0.99)

dadosFundoDummies.shape

def split_datasets(features, outcome, test_size = 0.2):
  X_train, X_test, y_train, y_test = train_test_split(features, outcome, test_size = test_size,
                                                      stratify = outcome, random_state = 0)
  y_train, y_test = np.array(y_train).ravel(), np.array(y_test).ravel()
  
  return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = split_datasets(dadosFundoDummies, resultadosFundo)

X_train.shape

X_test.shape

"""### Modelagem preliminar

AUC e ROC: https://medium.com/bio-data-blog/entenda-o-que-%C3%A9-auc-e-roc-nos-modelos-de-machine-learning-8191fb4df772

#### Random Forest Classifier
"""

def rfc_test(X_train, y_train, n_estimators = 100, cv = 10):
  np.random.seed(0)
  rfc = RandomForestClassifier(n_estimators = n_estimators, random_state = 0, n_jobs = -1)
  cv_scores = cross_val_score(rfc, X_train, y_train, scoring = 'roc_auc', cv = cv, n_jobs = -1)
  print('Média dos 10 testes Random Forest: ', cv_scores.mean())

rfc_test(X_train, y_train)

"""#### Gradient Boosting Classifer

Métodos com árvores de decisão: https://iaexpert.com.br/index.php/2019/04/18/xgboost-a-evolucao-das-arvores-de-decisao/

Gradient Boosting: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
"""

def gbc_test(X_train, y_train, n_estimators = 100, learning_rate = 0.1, cv = 5):
  np.random.seed(0)
  gbc = GradientBoostingClassifier(n_estimators = n_estimators, learning_rate = learning_rate,
                                  random_state = 0)
  cv_scores = cross_val_score(gbc, X_train, y_train, scoring = 'roc_auc', cv = cv, n_jobs = -1)
  print('Média dos 10 testes Gradient Boosting: ', cv_scores.mean())

gbc_test(X_train, y_train)




def optimize(n_estimators, learning_rate, min_samples_split, min_samples_leaf,
             max_depth, max_features, subsample, params, cv = 10):
  np.random.seed(0)
  gbc = GradientBoostingClassifier(n_estimators = n_estimators,
                                  learning_rate = learning_rate,
                                  min_samples_split = min_samples_split,
                                  min_samples_leaf = min_samples_leaf,
                                  max_depth = max_depth,
                                  max_features = max_features,
                                  subsample = subsample,
                                  random_state = 0)
  grid_search = GridSearchCV(estimator = gbc, param_grid = params, scoring = 'roc_auc',
                             n_jobs = -1, iid = False, cv = cv)
  grid_search.fit(X_train, y_train)
  results = grid_search.cv_results_
  best_params = grid_search.best_params_
  best_score = grid_search.best_score_
  print(best_params, best_score)
  
  return gbc, best_params, best_score

# Variáveis para coletar os resultados
modelsGBC = np.array([])
opt_paramsGBC = dict()
scoresGBC = np.array([])

# MODELO 0
learning_rate = 0.1
n_estimators = None
max_depth = 8
min_samples_split = 250
min_samples_leaf = 20
max_features = 'sqrt'
subsample = 0.8
params = {'n_estimators': range(50, 151, 10)}

gbc, opt_param, score = optimize(n_estimators = n_estimators,
                                 learning_rate = learning_rate,
                                 min_samples_split = min_samples_split,
                                 min_samples_leaf = min_samples_leaf,
                                 max_depth = max_depth,
                                 max_features = max_features,
                                 subsample = subsample,
                                 params = params)

# Atualizando as variáveis com os resultados
modelsGBC = np.append(modelsGBC, gbc)
opt_paramsGBC = {**opt_paramsGBC, **opt_param}
scoresGBC = np.append(scoresGBC, score)

"""### Otimizando `max_depth` e `min_samples_split`"""

# MODELO 1
n_estimators = opt_paramsGBC['n_estimators']
max_depth = None
min_samples_split = None
min_samples_leaf = 20
max_features = 'sqrt'
subsample = 0.8
params = {'max_depth': range(3, 12, 2), 'min_samples_split': range(150, 401, 50)}

gbc, opt_param, score = optimize(n_estimators = n_estimators,
                                 learning_rate = learning_rate,
                                 min_samples_split = min_samples_split,
                                 min_samples_leaf = min_samples_leaf,
                                 max_depth = max_depth,
                                 max_features = max_features,
                                 subsample = subsample,
                                 params = params)

# Atualizando as variáveis com os resultados
modelsGBC = np.append(modelsGBC, gbc)
opt_paramsGBC = {**opt_paramsGBC, **opt_param}
scoresGBC = np.append(scoresGBC, score)

"""### Otimizando `min_samples_leaf`"""

# MODELO 2
max_depth = opt_paramsGBC['max_depth']
min_samples_split = opt_paramsGBC['min_samples_split']
min_samples_leaf = None
max_features = 'sqrt'
subsample = 0.8
params = {'min_samples_leaf': range(25, 61, 5)}

gbc, opt_param, score = optimize(n_estimators = n_estimators,
                                 learning_rate = learning_rate,
                                 min_samples_split = min_samples_split,
                                 min_samples_leaf = min_samples_leaf,
                                 max_depth = max_depth,
                                 max_features = max_features,
                                 subsample = subsample,
                                 params = params)

# Atualizando as variáveis com os resultados
modelsGBC = np.append(modelsGBC, gbc)
opt_paramsGBC = {**opt_paramsGBC, **opt_param}
scoresGBC = np.append(scoresGBC, score)

"""### Otimizando `max_features`"""

# MODELO 3
min_samples_leaf = opt_paramsGBC['min_samples_leaf']
max_features = None
subsample = 0.8
params = {'max_features': range(21, 31, 1)}

gbc, opt_param, score = optimize(n_estimators = n_estimators,
                                 learning_rate = learning_rate,
                                 min_samples_split = min_samples_split,
                                 min_samples_leaf = min_samples_leaf,
                                 max_depth = max_depth,
                                 max_features = max_features,
                                 subsample = subsample,
                                 params = params)

# Atualizando as variáveis com os resultados
modelsGBC = np.append(modelsGBC, gbc)
opt_paramsGBC = {**opt_paramsGBC, **opt_param}
scoresGBC = np.append(scoresGBC, score)

"""### Otimizando `subsample`"""

max_features = opt_paramsGBC['max_features']
subsample = 1
params = {'subsample': np.append(np.arange(0.6, 1, 0.05), 1)}

np.append(np.arange(0.6, 1, 0.05), 1)

gbc, opt_param, score = optimize(n_estimators = n_estimators,
                                 learning_rate = learning_rate,
                                 min_samples_split = min_samples_split,
                                 min_samples_leaf = min_samples_leaf,
                                 max_depth = max_depth,
                                 max_features = max_features,
                                 subsample = subsample,
                                 params = params)

# Atualizando as variáveis com os resultados
modelsGBC = np.append(modelsGBC, gbc)
opt_paramsGBC = {**opt_paramsGBC, **opt_param}
scoresGBC = np.append(scoresGBC, score)


"""### Otimizando `n_estimators` e `learning_rate`"""

def optimize2(n_estimators, learning_rate, min_samples_split, min_samples_leaf,
             max_depth, max_features, subsample, cv = 5):
  np.random.seed(0)
  gbc = GradientBoostingClassifier(n_estimators = n_estimators,
                                  learning_rate = learning_rate,
                                  min_samples_split = min_samples_split,
                                  min_samples_leaf = min_samples_leaf,
                                  max_depth = max_depth,
                                  max_features = max_features,
                                  subsample = subsample,
                                  random_state = 0)
  cv_scores = cross_val_score(gbc, X_train, y_train, scoring = 'roc_auc', cv = cv, n_jobs = -1)
  score = cv_scores.mean()
  
  return gbc, score

learning_rate = 0.1
n_estimators = opt_paramsGBC['n_estimators']

# MODELO 5
subsample = opt_paramsGBC['subsample']
learning_rate /= 2
n_estimators *= 2

learning_rate, n_estimators

gbc, score = optimize2(n_estimators = n_estimators,
                       learning_rate = learning_rate,
                       min_samples_split = min_samples_split,
                       min_samples_leaf = min_samples_leaf,
                       max_depth = max_depth,
                       max_features = max_features,
                       subsample = subsample)

# Atualizando as variáveis com os resultados
modelsGBC = np.append(modelsGBC, gbc)
scoresGBC = np.append(scoresGBC, score)

# MODELO 6
learning_rate /= 5
n_estimators *= 5
learning_rate, n_estimators

gbc, score = optimize2(n_estimators = n_estimators,
                       learning_rate = learning_rate,
                       min_samples_split = min_samples_split,
                       min_samples_leaf = min_samples_leaf,
                       max_depth = max_depth,
                       max_features = max_features,
                       subsample = subsample)

# Atualizando as variáveis com os resultados
modelsGBC = np.append(modelsGBC, gbc)
scoresGBC = np.append(scoresGBC, score)

# MODELO 7
learning_rate /= 2
n_estimators *= 2
learning_rate, n_estimators

gbc, score = optimize2(n_estimators = n_estimators,
                       learning_rate = learning_rate,
                       min_samples_split = min_samples_split,
                       min_samples_leaf = min_samples_leaf,
                       max_depth = max_depth,
                       max_features = max_features,
                       subsample = subsample)

# Atualizando as variáveis com os resultados
modelsGBC = np.append(modelsGBC, gbc)
scoresGBC = np.append(scoresGBC, score)

# MODELO 8
learning_rate /= 5
n_estimators *= 5
learning_rate, n_estimators

gbc, score = optimize2(n_estimators = n_estimators,
                       learning_rate = learning_rate,
                       min_samples_split = min_samples_split,
                       min_samples_leaf = min_samples_leaf,
                       max_depth = max_depth,
                       max_features = max_features,
                       subsample = subsample)

# Atualizando as variáveis com os resultados
modelsGBC = np.append(modelsGBC, gbc)
scoresGBC = np.append(scoresGBC, score)
types = X_train.get_dtype_counts()

plt.plot(scoresGBC)
plt.title('Modelos testados para o algoritmo GBC')
plt.xlabel('Modelo')
plt.ylabel('AUC')
plt.show()

print('Modelo GBC de melhor desempenho: ', scoresGBC.argmax(), ' - AUC: ', max(scoresGBC))

best_model = modelsGBC[scoresGBC.argmax()]

best_model.fit(X_train, y_train)

y_pred_probs = best_model.predict_proba(X_test)

y_pred_probs

y_pred_probs[:, 1]

print('Modelo GBC de melhor desempenho rodado na base de teste: ', roc_auc_score(y_test, y_pred_probs[:, 1]))


def optimizeRF(n_estimators, min_samples_split, min_samples_leaf,
             max_depth, max_features, bootstrap, params, cv = 10):
  np.random.seed(0)
  
  rfc = RandomForestClassifier(n_estimators = n_estimators,
                                  min_samples_split = min_samples_split,
                                  min_samples_leaf = min_samples_leaf,
                                  max_depth = max_depth,
                                  max_features = max_features,
                                  bootstrap = bootstrap,
                                  random_state = 0)
  grid_search = GridSearchCV(estimator = rfc, param_grid = params, scoring = 'roc_auc',
                             n_jobs = -1, iid = False, cv = cv)
  grid_search.fit(X_train, y_train)
  results = grid_search.cv_results_
  best_params = grid_search.best_params_
  best_score = grid_search.best_score_
  print(best_params, best_score)
  
  return rfc, best_params, best_score

# Variáveis para coletar os resultados
modelsRFC = np.array([])
opt_paramsRFC = dict()
scoresRFC = np.array([])

# MODELO 0
n_estimators = None
max_depth = 8
min_samples_split = 250
min_samples_leaf = 20
max_features = 'sqrt'
bootstrap = True
params = {'n_estimators': range(50, 151, 10)}

rfc, opt_param, score = optimizeRF(n_estimators = n_estimators,
                                 min_samples_split = min_samples_split,
                                 min_samples_leaf = min_samples_leaf,
                                 max_depth = max_depth,
                                 max_features = max_features,
                                 bootstrap = bootstrap,
                                 params = params)

# Atualizando as variáveis com os resultados
modelsRFC = np.append(modelsRFC, rfc)
opt_paramsRFC = {**opt_paramsRFC, **opt_param}
scoresRFC = np.append(scoresRFC, score)

"""### Otimizando `max_depth` e `min_samples_split`"""

# MODELO 1
n_estimators = opt_paramsRFC['n_estimators']
max_depth = None
min_samples_split = None
min_samples_leaf = 20
max_features = 'sqrt'
bootstrap = True
params = {'max_depth': range(3, 12, 2), 'min_samples_split': range(150, 401, 50)}

rfc, opt_param, score = optimizeRF(n_estimators = n_estimators,
                                 min_samples_split = min_samples_split,
                                 min_samples_leaf = min_samples_leaf,
                                 max_depth = max_depth,
                                 max_features = max_features,
                                 bootstrap = bootstrap,
                                 params = params)

# Atualizando as variáveis com os resultados
modelsRFC = np.append(modelsRFC, rfc)
opt_paramsRFC = {**opt_paramsRFC, **opt_param}
scoresRFC = np.append(scoresRFC, score)

"""### Otimizando `min_samples_leaf`"""

# MODELO 2
max_depth = opt_paramsRFC['max_depth']
min_samples_split = opt_paramsRFC['min_samples_split']
min_samples_leaf = None
max_features = 'sqrt'
bootstrap = True
params = {'min_samples_leaf': range(25, 61, 5)}

rfc, opt_param, score = optimizeRF(n_estimators = n_estimators,
                                 min_samples_split = min_samples_split,
                                 min_samples_leaf = min_samples_leaf,
                                 max_depth = max_depth,
                                 max_features = max_features,
                                 bootstrap = bootstrap,
                                 params = params)

# Atualizando as variáveis com os resultados
modelsRFC = np.append(modelsRFC, rfc)
opt_paramsRFC = {**opt_paramsRFC, **opt_param}
scoresRFC = np.append(scoresRFC, score)

"""### Otimizando `max_features`"""

# MODELO 3
min_samples_leaf = opt_paramsRFC['min_samples_leaf']
max_features = None
bootstrap = True
params = {'max_features': range(21, 31, 1)}

rfc, opt_param, score = optimizeRF(n_estimators = n_estimators,
                                 min_samples_split = min_samples_split,
                                 min_samples_leaf = min_samples_leaf,
                                 max_depth = max_depth,
                                 max_features = max_features,
                                 bootstrap = bootstrap,
                                 params = params)

# Atualizando as variáveis com os resultados
modelsRFC = np.append(modelsRFC, rfc)
opt_paramsRFC = {**opt_paramsRFC, **opt_param}
scoresRFC = np.append(scoresRFC, score)

"""### Otimizando `bootstrap`"""

max_features = opt_paramsRFC['max_features']
bootstrap = None
params = {'bootstrap': (False, True)}

np.append(np.arange(0.6, 1, 0.05), 1)

rfc, opt_param, score = optimizeRF(n_estimators = n_estimators,
                                 min_samples_split = min_samples_split,
                                 min_samples_leaf = min_samples_leaf,
                                 max_depth = max_depth,
                                 max_features = max_features,
                                 bootstrap = bootstrap,
                                 params = params)

# Atualizando as variáveis com os resultados
modelsRFC = np.append(modelsRFC, rfc)
opt_paramsRFC = {**opt_paramsRFC, **opt_param}
scoresRFC = np.append(scoresRFC, score)

plt.plot(scoresRFC)
plt.title('Modelos testados para o algoritmo Random Forest')
plt.xlabel('Modelo')
plt.ylabel('AUC')
plt.show()

print('Modelo Random Forest de melhor desempenho: ', scoresRFC.argmax(), ' - AUC: ', max(scoresRFC))

best_model = modelsRFC[scoresRFC.argmax()]

X_train.dtypes
X_train.isna()
dados_vaziosXColuna = X_train.isna().sum(axis = 0)
dados_vaziosXColuna
X_trainInt = X_train.astype('int')

best_model.fit(X_train, y_train)

y_pred_probs = best_model.predict_proba(X_test)

y_pred_probs

y_pred_probs[:, 1]


# roc_auc_score(y_test, y_pred_probs[:, 1])

print('Modelo RF de melhor desempenho rodado na base de teste: ', roc_auc_score(y_test, y_pred_probs[:, 1]))


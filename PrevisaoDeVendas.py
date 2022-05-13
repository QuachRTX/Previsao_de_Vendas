# Modelo de Previsão de Vendas com base no investimento em anuncios

import pandas as pd # Para utilizar tabelas
import seaborn as sns # Para criar graficos
import matplotlib.pyplot as plt # Para criar IAs


tabela = pd.read_csv(r"C:\Users\User\basededados.csv")
#print(tabela)

#----------------------------------------------------------------------------------------------------------------------

sns.heatmap(tabela.corr(), annot=True, cmap="Wistia") # cria grafico
#plt.show()  # exibe o grafico

#----------------------------------------------------------------------------------------------------------------------

y = tabela["Vendas"] # Separo as colunas da tabela e atribuo em variaveis
x = tabela[["TV", "Radio", "Jornal"]]

from sklearn.model_selection import train_test_split # importa o teste da biblioteca

x_treino, x_teste, y_treino, y_teste = train_test_split(x,y, test_size=0.3, random_state=1) # Separa em 4 partes a tabela -> test_size define a % do teste

from sklearn.linear_model import LinearRegression # Importa o Modelo de IA 1
from sklearn.ensemble import RandomForestRegressor # Importa o Modelo de IA 2

modelo_regressaolinear = LinearRegression()  # Declaração do Modelo IA 1 em uma variavel
modelo_arvoredecisao = RandomForestRegressor()  # Declaração do Modelo IA 2 em uma vareavel

modelo_regressaolinear.fit(x_treino, y_treino) # Atribuindo os valores que serão usados na IA 1
modelo_arvoredecisao.fit(x_treino, y_treino)    # Atribuindo os valores que serão usados na IA 2

previsao_regressaolinear = modelo_regressaolinear.predict(x_teste)  # Efetua a predicao dos valores de teste dados para a IA 1
previsao_arvoredecisao = modelo_arvoredecisao.predict(x_teste)  # Efetua a predicao dos valores de teste dados para a IA 2

from sklearn.metrics import r2_score

#print(r2_score(y_teste, previsao_regressaolinear)) # Pega o resultado CORRETO dos valores de teste utilizados e compara com a predicao feita pela IA 1 e retorna em % o acerto
#print(r2_score(y_teste, previsao_arvoredecisao))  # Pega o resultado CORRETO dos valores de teste utilizados e compara com a predicao feita pela IA 2 e retorna em % o acerto

#----------------------------------------------------------------------------------------------------------------------

tabela_auxiliar = pd.DataFrame() # cria nova tabela
tabela_auxiliar["y teste"] = y_teste # cria coluna
tabela_auxiliar["Previsao Regressao Linear"] = previsao_regressaolinear
tabela_auxiliar["Previsao Arvore Decisao"] = previsao_arvoredecisao


plt.figure(figsize=(15,6)) # Aumenta o tamanho da figura
sns.lineplot(data=tabela_auxiliar)
#plt.show()

#----------------------------------------------------------------------------------------------------------------------

novos = pd.read_csv(r"C:\Users\User\valoresparaprevisao.csv") # Importa nova tabela de valores


previsao = modelo_arvoredecisao.predict(novos) # Utiliza a IA com melhor previsao para prever o numero de vendas de novos valores
novos["Previsao de Vendas"] = previsao # Em milhões
print(novos)





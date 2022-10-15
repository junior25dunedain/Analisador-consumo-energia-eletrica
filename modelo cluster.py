import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pylab
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import warnings
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist, pdist

warnings.filterwarnings('ignore')

# avaliador do consumo de energia eletrica por um modelo de clusterização

# carregando os dados
dados = pd.read_csv('consumo_energia.txt',delimiter=';')

print(dados.info(),'\n')
print(dados.head())

# checando valores missing
print(dados.isna().sum())
# remover os valores NAT e retirar as duas primeiras colunas que não são importantes para a minha aplicação
dados = dados.iloc[:,2:].dropna()

# convertendo todos os daos
df = pd.DataFrame(dados,dtype='float64')
print(df.info(),'\n')
print(df.isnull().values.any())
df_valores = df.values

#separando os dados de treino e teste
amostra1 , amostra2 = train_test_split(df_valores,train_size=0.015,random_state=42)
print('\n',amostra1)


# usando o PCA para diminuir a dimensionalidade do conjunto de dados
# transformando as 7 variaveis em 2 componentes principais.
pca = PCA(n_components=2).fit_transform(amostra1)



# Realização de uma analise para encontrar o modelo cluster mais adequado para a minha aplicação
# determinando um range de k
'''
k_range = range(1,12)

# criando uma lista de modelos KMeans e treinando
K_means_modelo = [KMeans(n_clusters= k).fit(pca) for k in k_range]

# ajustando os modelos com o metodo centroide
centriodes = [i.cluster_centers_ for i in K_means_modelo]

# calculando a distancia euclidiana
k_euclidiana = [cdist(pca,cent,'euclidean') for cent in centriodes]
dist = [np.min(ke,axis=1) for ke in k_euclidiana]

#soma dos quadrados das distancias dentro do cluster
soma_quadrados_intra_cluster = [sum(d**2) for d in dist]

#soma total
soma_total = sum(pdist(pca)**2)/pca.shape[0]

# soma dos quadrados entre clusters
soma_quadrados_inter_cluster = soma_total - soma_quadrados_intra_cluster

# gerando o grafico da curva de Elbow, para encontrar o modelo cluster com o numero de agrupamentos mais adequado para a aplicação
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(k_range,soma_quadrados_inter_cluster/soma_total *100,'b*-')
ax.set_ylim((0,100))
plt.grid(True)
plt.xlabel('Número de Clusters')
plt.ylabel('Percentual de Variância explicda')
plt.title('Variância explicda x valor de k')
plt.show()
'''

# O modelo cluster ideal é com 8 clusters
modelo_cluster_final = KMeans(n_clusters=8)
modelo_cluster_final.fit(pca)

# obtém os valores minimos e máximos, e organiza o shape
x_min, x_max = pca[:,0].min() - 5, pca[:,0].max()-1
y_min, y_max = pca[:,1].min() - 1, pca[:,1].max()-5

xx,yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min,y_max,0.02))
Z = modelo_cluster_final.predict(np.c_[xx.ravel(),yy.ravel()])
Z = Z.reshape(xx.shape)

# plot das areas dos clusters
plt.figure(1)
plt.clf()
plt.imshow(Z,interpolation='nearest',extent = (xx.min(),xx.max(),yy.min(),yy.max()),cmap=plt.cm.Paired,aspect='auto',origin='lower')
plt.show()

#plot dos centroides
plt.plot(pca[:,0],pca[:,1],'k.',markersize=4)
centriods = modelo_cluster_final.cluster_centers_
inert = modelo_cluster_final.inertia_
plt.scatter(centriods[:,0],centriods[:,1], marker='x',s=169,linewidth=3,color='r',zorder=8)
plt.xlim(x_min,x_max)
plt.ylim(y_min,y_max)
plt.xticks(())
plt.yticks(())
plt.show()


# avaliando a clusterização por meio do silhouette_score
labels = modelo_cluster_final.labels_
print('O Desempenho do modelo_cluster_final é ',round(silhouette_score(pca,labels,metric='euclidean'),3),'\n')

# Criando outro modelo cluster com 10 clusters
modelo_cluster2 = KMeans(n_clusters=10)
modelo_cluster2.fit(pca)

# obtém os valores minimos e máximos, e organiza o shape

Z2 = modelo_cluster2.predict(np.c_[xx.ravel(),yy.ravel()])
Z2 = Z2.reshape(xx.shape)

# plot das areas dos clusters2
plt.figure(1)
plt.clf()
plt.imshow(Z2,interpolation='nearest',extent = (xx.min(),xx.max(),yy.min(),yy.max()),cmap=plt.cm.Paired,aspect='auto',origin='lower')
plt.show()

#plot dos centroides2
plt.plot(pca[:,0],pca[:,1],'k.',markersize=4)
centriods = modelo_cluster2.cluster_centers_
inert = modelo_cluster2.inertia_
plt.scatter(centriods[:,0],centriods[:,1], marker='x',s=169,linewidth=3,color='r',zorder=8)
plt.xlim(x_min,x_max)
plt.ylim(y_min,y_max)
plt.xticks(())
plt.yticks(())
plt.show()

# avaliando a clusterização por meio do silhouette_score
labels2 = modelo_cluster2.labels_
print('O Desempenho do modelo_cluster2 é ',round(silhouette_score(pca,labels2,metric='euclidean'),3),'\n')


# criando o cluster map com os dados do modelo_cluster_final que apresentou melhor resultado
cluster_map = pd.DataFrame(amostra1, columns= df.columns)
cluster_map['Global_active_power'] = pd.to_numeric(cluster_map['Global_active_power'])
# criando uma nova coluna que corresponde aos dados de saida do modelo cluster_cluster_final
cluster_map['cluster'] = modelo_cluster_final.labels_

print(cluster_map.sample(6),'\n')

# calculando a media de potencia consumida por cada grupo de consumidores
print(cluster_map.groupby('cluster')['Global_active_power'].mean())

# calculando a quantidade de elementos por grupo
print(cluster_map.groupby('cluster')['Global_active_power'].count())


#!/usr/bin/env python
# coding: utf-8

# In[1]:


# tabella dati mancanti
import pandas as pd

# Dataset con dati mancanti rappresentati da None o NaN
dataset = [
    {"età": 25, "punteggio": 90, "ammesso": 1},
    {"età": None, "punteggio": 85, "ammesso": 0},
    {"età": 28, "punteggio": None, "ammesso": 1},
    {"età": None, "punteggio": 75, "ammesso": 1},
    {"età": 23, "punteggio": None, "ammesso": None},
    {"età": 23, "punteggio": 77, "ammesso": None},
]
df = pd.DataFrame(dataset)
df


# In[2]:


# identificazione delle righe con dati mancanti
righe_con_dati_mancanti =df[df.isnull().any(axis=1)]# da df in poi scpiega le righe con i dati mancanti 
righe_con_dati_mancanti


# In[7]:


#conta quante righe con dati mancanti ci sono totale
totale_dati_mancanti = righe_con_dati_mancanti.shape[0]# con shape 0 da ci da il  numero delle righe hanno almeno un elemento mancante
# shape 1 ci da le colonne
totale_dati_mancanti


# In[4]:


import pandas as pd

# Dataset con dati mancanti rappresentati da None o NaN
dataset = [
    {"nome": "Alice", "età": 25, "punteggio": 90, "email": "alice@email.com"},
    {"nome": "Bob", "età": 22, "punteggio": None, "email": None},
    {"nome": "Charlie", "età": 28, "punteggio": 75, "email": "charlie@email.com"},
]

# Converti il dataset in un DataFrame
df = pd.DataFrame(dataset)
df


# In[14]:


#rimuovi le righe con dati mancanti
df1=df.dropna(inplace=False)
df1


# In[14]:


# riempimento con la media dei dati
df1[numeri_cols.columns] = df[numeri_cols.columns].fillna(df[numeri_cols.columns].mean().iloc[0])
df1 # fillna = vuole dire riempi NAN quindi, con la media


# In[13]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# Genera dati di esempio
data = {
    'Variable1': [1, 2, 3, 4, 5],
    'Variable2': [1, 2, np.nan, 4, np.nan],
    'Missing_Column': ['A', 'B', 'A', 'C', np.nan]
}
# Crea un DataFrame
df = pd.DataFrame(data)
df1=pd.DataFrame()
df


# In[4]:


numeric_cols =df.select_dtypes(include=['numbers'])
numeric_cols.columns# .colums da il nome delle colonne del dataframe


# In[5]:


categorical_cols =df.select_dtypes(exclude=['numbers'])
categorical_cols


# In[7]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Genera dati di esempio
data = {
    'Feature1': [1, 2, np.nan, 4, 5],
    'Feature2': [np.nan, 2, 3, 4, np.nan],
    'Feature3': [1, np.nan, 3, 4, 5]
}
# Crea un DataFrame
df = pd.DataFrame(data)
df


# In[8]:


df.isnull()# ricrea un data frame che quando ci sono degli elementi nulli mette true


# In[9]:


df.isnull().sum()# .sum fa lasomma per colonne


# In[10]:


missing_percent = (df.isnull().sum / len(df)) * 100
missing_percent


# In[11]:


# create un DataDFrame
df = pd.DataFrame(data)
df1=pd.DataFrame()
df


# In[12]:


import seaborn as sns


# In[2]:


pip install seaborn


# In[16]:


plt.figure(figsize=(8, 6))
snsheatmap(missing_mattrix, cmap='viridis', cbar=false,alpha=0.8)
plt.title('matrice di missimg Values')
plt.show()


# In[9]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Genera dati di esempio
data = {
    'Feature1': [1, 2, np.nan, 4, 5],
    'Feature2': [np.nan, 2, 3, 4, np.nan],
    'Feature3': [1, np.nan, 3, 4, 5]
}
# Crea un DataFrame
df = pd.DataFrame(data)
df


# In[10]:


df.isnull()


# In[11]:


df.isnull().sum()


# In[6]:


missing_percent = (df.isnull().sum() / len(df))*100
missing_percent


# In[ ]:





# In[12]:


missing_percent = (df.isnull().sum() / len(df)) * 100

plt.figure(figsize=(10, 6))
missing_percent.plot(kind='bar', color='skyblue',alpha=0.8)
plt.xlabel('variabili')
plt.ylabel('analisi dei missing values per variabile')
plt.xticks(rotation=0)
plt.show()


# In[13]:


plt.figure(figsize=(8, 6))
sns.heatmap(missing_matrix, cmap='viridis', cbar=False,alpha=0.8)
plt.title('Matrice di missing values')
plt.show 


# In[14]:


pip install plotly


# In[17]:


pip install pandas


# In[22]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Genera dati casuali per l'esplorazione
np.random.seed(2)
data = {
    'Età': np.random.randint(18, 70, size=1000),
    'Genere': np.random.choice(['Maschio', 'Femmina'], size=1000),
    'Punteggio': np.random.uniform(0, 100, size=1000),
    'Reddito': np.random.normal(50000, 15000, size=1000)
}

df = pd.DataFrame(data)

# Visualizza le prime righe del dataset
print(df.head())import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Genera dati di esempio
data = {
    'Feature1': [1, 2, np.nan, 4, 5],
    'Feature2': [np.nan, 2, 3, 4, np.nan],
    'Feature3': [1, np.nan, 3, 4, 5]
}

# Crea un DataFrame
df = pd.DataFrame(data)

# Calcola la matrice di missing values
missing_matrix = df.isnull()
missing_matrix


# In[15]:


print(df.info())

print(df.describe())


# In[15]:


pip install seaborn


# In[20]:


plt.figure(figsize=(12,6))
sns.set_style("whitegrid")
sns.hitsplot(df["Reddito"], kde=False, bins=50, label="Reddito")
plt.legend()
plt.title('Distribuzione delle variabili numeriche')
plt.show()


# In[19]:


numeric_features = df.select_dtypes(include=[np.number])
sns.pairplot(df[numeric_features.columns])
plt.title('Matrice di Scatter Plot tra variabili numeriche')
plt.show()


# In[31]:


plt.figure(figsize=(10, 6))
sns.boxplot(x='Genere',y='Punteggio')
plt.title('box plot Genere maschile e femminile')
plt.show()


# In[32]:


import plotly.express as px
fig=px.scatter(df, x='Età', y='Reddito', color='Genere', size='Punteggio')
fig.update_layout(title='Grafico a dispersione interattivo')
fig.show()


# In[ ]:





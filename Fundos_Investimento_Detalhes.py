#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
import pyodbc 


# In[2]:

# Função que executa o download dos dados detalhados do fundo de investimento a partir de seu CNPJ
def carregaDadosFundo(cnpj):
    sessao = requests.session()
    headers = {
    'Host': 'maisretorno.com',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:72.0) Gecko/20100101 Firefox/72.0',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'pt-BR,pt;q=0.8,en-US;q=0.5,en;q=0.3',
    'Accept-Encoding': 'gzip, deflate, br',
    'Authorization': 'Basic YXBpOlIkX1hKZk1uNVdhaHlKaA==',
    'Connection': 'keep-alive',
    'Referer': 'https://maisretorno.com/comparacao/principal/otimo/cdi/' + '{:014d}'.format(cnpj),
    'TE': 'Trailers'
    }

    pageContent=sessao.get(
         url = 'https://maisretorno.com/api/v1/fundos/search/' + '{:014d}'.format(cnpj) + '/', headers=headers
    )
    #print('https://maisretorno.com/api/v1/fundos/search/' + '{:014d}'.format(cnpj) + '/')
    #print(pageContent.content)
    respJSON = pageContent.json()
    for fundo in respJSON:
        if fundo['c']==cnpj:
            pageContent=sessao.get(
                 url = 'https://maisretorno.com/api/v1/fundos/get/' + fundo['s'] + '/', headers=headers
            )
            #print(pageContent.content)
            respJSON = pageContent.json()
    # print(str(cnpj) + ' ' + respJSON['fundo']['nome'])
    return respJSON
    


# In[3]:

# Função que insere os dados do fundo em banco de dados SQL Server
def carregaDadosFundoBanco(respJSON):
    cursor = cnxn.cursor()

    if respJSON['admin']['cnpj']!=None:
        cursor.execute("""SELECT [cnpj] FROM  [Fundos_05052020].[administradores] Where cnpj = ?""", respJSON['admin']['cnpj'])
        if cursor.fetchone() is None:
            sql = """INSERT INTO [Fundos_05052020].[administradores]
                       ([cnpj]
                       ,[nome]
                       ,[slug]
                       )    
                       VALUES (?, ?, ?)"""
            cursor.execute(sql, respJSON['admin']['cnpj'], respJSON['admin']['nome'], respJSON['admin']['slug'])

    sql = """INSERT INTO [Fundos_05052020].[fundos_dados]
               ([cnpj]
               ,[nome]
               ,[nome_abreviado]
               ,[de_cotas]
               ,[benchmark]
               ,[cond_aberto]
               ,[prazo_pagamento]
               ,[unique_slug]
               ,[patrimonio]
               ,[cotistas]
               ,[last_update]
               ,[last_quote_date]
               ,[first_quote_date]
               ,[exclusivo]
               ,[restrito]
               ,[invest_qualif]
               ,[tipo_de_previdencia]
               ,[tribut_lp]
               ,[situacao]
               ,[status]
               ,[tipo_de_fundo]
               ,[indice]
               ,[cnpjAdmin]
               ,[classe_p_n]
               ,[classe_p_s]
               ,[classe_c_n]
               ,[classe_c_s])    
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
    cursor.execute(sql, respJSON['fundo']['cnpj'], respJSON['fundo']['nome'], respJSON['fundo']['nome_abreviado'], respJSON['fundo']['de_cotas'], 
       respJSON['fundo']['benchmark'], respJSON['fundo']['cond_aberto'], respJSON['fundo']['prazo_pagamento'], 
        respJSON['fundo']['unique_slug'], respJSON['fundo']['patrimonio'], respJSON['fundo']['cotistas'], 
        respJSON['fundo']['last_update'], respJSON['fundo']['last_quote_date'], respJSON['fundo']['first_quote_date'], 
        respJSON['fundo']['exclusivo'], respJSON['fundo']['restrito'], respJSON['fundo']['invest_qualif'], 
        respJSON['fundo']['tipo_de_previdencia'], respJSON['fundo']['tribut_lp'], respJSON['fundo']['situacao'], 
        respJSON['fundo']['status'], respJSON['fundo']['tipo_de_fundo'], respJSON['fundo']['index'], 
        respJSON['admin']['cnpj'], respJSON['class']['p']['n'], respJSON['class']['p']['s'], 
        respJSON['class']['c']['n'], respJSON['class']['c']['s'])
    
    for gestor in respJSON['gestores']:
        cursor.execute("""SELECT [cnpj] FROM  [Fundos_05052020].[gestores] Where cnpj = ?""", gestor['cnpj'])
        if cursor.fetchone() is None:
            sql = """INSERT INTO [Fundos_05052020].[gestores]
                   ([cnpj]
                   ,[nome]
                   ,[slug]
                   )    
                   VALUES (?, ?, ?)"""
            cursor.execute(sql, gestor['cnpj'], gestor['nome'], gestor['slug'])

        sql = """INSERT INTO [Fundos_05052020].[fundosXgestores]
               ([cnpjGestor]
               ,[cnpjFundo]
               )    
               VALUES (?, ?)"""
        cursor.execute(sql, gestor['cnpj'], respJSON['fundo']['cnpj'])
    cursor.close()


# In[4]:

# Função que percorrea o dataset se fundos e executa o download dos dados para cada um deles.
def preencheDetalhesFundos():
    cursor = cnxn.cursor()
    cursor.execute("""SELECT cnpj
          FROM [Fundos_05052020].[Fundos] Where cnpj Not In (Select cnpj from [Fundos_05052020].fundos_dados)""")
    i = 1
    
    for fundo in cursor.fetchall():
        print(i)
        print(fundo.cnpj)
        respJson = carregaDadosFundo(fundo.cnpj)
        # print(respJson)
        carregaDadosFundoBanco(respJson)
        i = i + 1
        #if i==31:
        #    break
        
    cursor.close()


# In[20]:


cnpj = 7046169000121
respJson = carregaDadosFundo(cnpj)
print(respJson)


# In[6]:


cnxn = pyodbc.connect("Driver={SQL Server Native Client 11.0};"
                      "Server=localhost\\NOVAINSTANCIA2;"
                      "Database=Fundos_V2020;"
                      "Trusted_Connection=yes;")
cnxn.autocommit = True
#cursor = cnxn.cursor()
#cursor.execute("Truncate Table [Fundos_05052020].[fundosXgestores]")
#cursor.execute("Delete From [Fundos_05052020].gestores")
#cursor.execute("Delete From [Fundos_05052020].[fundos_dados]")
#cursor.execute("Delete From [Fundos_05052020].administradores")
#cursor.close()

preencheDetalhesFundos()
# carregaDadosFundoBanco(respJson)
cnxn.close()    


# In[ ]:





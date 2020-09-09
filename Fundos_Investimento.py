#!/usr/bin/env python
# coding: utf-8

# In[3]:


import requests
import datetime
import pyodbc 

sessao = requests.session()


# In[56]:


def carregaRentabilidadesFundo(cnpj):
    pageContent=sessao.get(
         'https://assets-comparacaodefundos.s3-sa-east-1.amazonaws.com/cvm/' + '{:014d}'.format(cnpj)
    )
    print('{:014d}'.format(cnpj))

    cursor = cnxn.cursor()
    # cursor.execute("Truncate Table [Fundos_05052020].[indicesDiarios]")
    # cursor.execute("Truncate Table [Fundos_05052020].[rentabilidadeMensal]")
    # cursor.execute("Truncate Table [Fundos_05052020].[rentabilidadeAnual]")

    if '<Error>' not in pageContent.text and pageContent.text!='':
        respJSON = pageContent.json()
        cotacoesMensais = []
        cotacoesAnuais = []
        if len(respJSON)>1:
            cotacaoAnterior = respJSON[0]
            i = 1
            for cotacao in respJSON[1:]:
                if cotacao['c'] > 0:
                    if cotacaoAnterior['c'] == 0:
                        cotacaoAnterior = cotacao
                    dataCotacaoAnterior = datetime.datetime.strptime(str(cotacaoAnterior['d']), '%Y%m%d')
                    dataCotacao = datetime.datetime.strptime(str(cotacao['d']), '%Y%m%d')
                    if dataCotacaoAnterior.month!=dataCotacao.month:
                        cotacoesMensais.append(cotacaoAnterior)
                    if dataCotacaoAnterior.year!=dataCotacao.year:
                        cotacoesAnuais.append(cotacaoAnterior)
                    #print(cotacao['c'])
                    #print(cotacao['d'])
                    #print(datetime.datetime.strptime(str(cotacao['d']), '%Y%m%d'))
                    #print(cotacao['p'])
                    #print(cotacao['q'])
                    if 'p' in cotacao:
                        patrimonio = cotacao['p']
                    else:
                        patrimonio = 0
                    if 'q' in cotacao:
                        quotas = cotacao['q']
                    else:
                        quotas = 0
                    cursorRent = cnxn.cursor()
                    cursorRent.execute("""SELECT [cnpj]
                          FROM [Fundos_05052020].[indicesDiarios] Where cnpj = ? And data=?""", cnpj, datetime.datetime.strptime(str(cotacao['d']), '%Y%m%d'))
                    if cursorRent.fetchone() is not None:
                        continue
                    sql = """INSERT INTO [Fundos_05052020].[indicesDiarios]
                           ([cnpj]
                           ,[data]
                           ,[rentabilidade]
                           ,[patrimonio]
                           ,[q])
                        VALUES (?, ?, ?, ?, ?)"""
                    cursor.execute(sql, cnpj, datetime.datetime.strptime(str(cotacao['d']), '%Y%m%d'), 
                                  cotacao['c'], patrimonio, quotas)

                    cotacaoAnterior = cotacao
                    i = i + 1
        # print(cotacoesMensais)
        if len(cotacoesMensais)>1:
            cotacaoAnterior = cotacoesMensais[0]
            for cotacao in cotacoesMensais[1:]:
                #print(cotacao['d'])
                #print("{:.2%}".format(cotacao['c']/cotacaoAnterior['c']-1))
                cursorRent = cnxn.cursor()
                if 'p' in cotacao:
                    patrimonio = cotacao['p']
                else:
                    patrimonio = 0
                if 'q' in cotacao:
                    quotas = cotacao['q']
                else:
                    quotas = 0
                cursorRent.execute("""SELECT [cnpj]
                      FROM [Fundos_05052020].[rentabilidadeMensal] Where cnpj = ? And data=?""", cnpj, datetime.datetime.strptime(str(cotacao['d']), '%Y%m%d'))
                if cursorRent.fetchone() is not None:
                    continue
                sql = """INSERT INTO [Fundos_05052020].[rentabilidadeMensal]
                       ([cnpj]
                       ,[data]
                       ,[indice]
                       ,[patrimonio]
                       ,[q]
                       ,[rentabilidade])
                    VALUES (?, ?, ?, ?, ?, ?)"""
                cursor.execute(sql, cnpj, datetime.datetime.strptime(str(cotacao['d']), '%Y%m%d'), 
                              cotacao['c'], patrimonio, quotas, cotacao['c']/cotacaoAnterior['c']-1)
                cotacaoAnterior = cotacao
        # print(cotacoesAnuais)
        if len(cotacoesAnuais)>1:
            cotacaoAnterior = cotacoesAnuais[0]
            for cotacao in cotacoesAnuais[1:]:
                if 'p' in cotacao:
                    patrimonio = cotacao['p']
                else:
                    patrimonio = 0
                if 'q' in cotacao:
                    quotas = cotacao['q']
                else:
                    quotas = 0
                # print(cotacao['d'])
                # print("{:.2%}".format(cotacao['c']/cotacaoAnterior['c']-1))
                cursorRent = cnxn.cursor()
                cursorRent.execute("""SELECT [cnpj]
                      FROM [Fundos_05052020].[rentabilidadeAnual] Where cnpj = ? And data=?""", cnpj, datetime.datetime.strptime(str(cotacao['d']), '%Y%m%d'))
                if cursorRent.fetchone() is not None:
                    continue
                sql = """INSERT INTO [Fundos_05052020].[rentabilidadeAnual]
                       ([cnpj]
                       ,[data]
                       ,[indice]
                       ,[patrimonio]
                       ,[q]
                       ,[rentabilidade])
                    VALUES (?, ?, ?, ?, ?, ?)"""
                cursor.execute(sql, cnpj, datetime.datetime.strptime(str(cotacao['d']), '%Y%m%d'), 
                              cotacao['c'], patrimonio, quotas, cotacao['c']/cotacaoAnterior['c']-1)
                cotacaoAnterior = cotacao
    else:
        print('Erro no CNPJ: ' + '{:014d}'.format(cnpj))
    # cursor.commit()
    cursor.close()


# In[57]:


def carregaFundos():
    pageContent=sessao.get(
         'https://assets-comparacaodefundos.s3-sa-east-1.amazonaws.com/cvm/fundos'
    )
    cursor = cnxn.cursor()
    # cursor.execute("Truncate Table [Fundos_05052020].[Fundos]")
    i = 1
    respJSON = pageContent.json()
    for fundo in respJSON:
        cnpj = fundo['c']
        if fundo['a']!='':
            data = datetime.datetime.strptime(str(fundo['a']), '%Y-%m-%d')
        else:
            data = None
        classe = fundo['g']
        nome = fundo['n']
        patrimonio = fundo['p']
        cotistas = fundo['q']
        iVar = fundo['i']
        tVar = fundo['t']

        cursorRent = cnxn.cursor()
        cursorRent.execute("""SELECT [cnpj]
              FROM [Fundos_05052020].[Fundos] Where cnpj = ?""", cnpj)
        if cursorRent.fetchone() is not None:
            continue
        sql = """INSERT INTO [Fundos_05052020].[Fundos]
           ([cnpj]
           ,[classe]
           ,[data]
           ,[nome]
           ,[patrimonio]
           ,[cotistas]
           ,[i]
           ,[t])
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)"""
        cursor.execute(sql, cnpj, classe, data, 
                      nome, patrimonio, cotistas, iVar, tVar)

        i = i + 1
        #if i==1000:
        #    break

        
    # cursor.commit()
    cursor.close()


# In[58]:


def preencheRentabilidadesFundos():
    cursor = cnxn.cursor()
    cursor.execute("""SELECT *
          FROM [Fundos_05052020].[Fundos] Where cnpj Not In (SELECT Distinct cnpj
  FROM [Fundos].[Fundos_05052020].[indicesDiarios])""")
    i = 1
    
    for fundo in cursor.fetchall():
        carregaRentabilidadesFundo(fundo.cnpj)

        i = i + 1
        #if i==10:
        #    break

        
    # cursor.commit()
    cursor.close()


# In[60]:


cnxn = pyodbc.connect("Driver={SQL Server Native Client 11.0};"
                      "Server=localhost\\NOVAINSTANCIA2;"
                      "Database=Fundos;"
                      "Trusted_Connection=yes;")
cnxn.autocommit = True

cursor = cnxn.cursor()
# cursor.execute("Truncate Table [Fundos_05052020].[Fundos]")
# cursor.execute("Truncate Table [Fundos_05052020].[indicesDiarios]")
# cursor.execute("Truncate Table [Fundos_05052020].[rentabilidadeMensal]")
# cursor.execute("Truncate Table [Fundos_05052020].[rentabilidadeAnual]")
# carregaFundos()
preencheRentabilidadesFundos()
cnxn.close()


# In[15]:


def carregaRentabilidadesCDI():
    pageContent=sessao.get(
         'https://assets-comparacaodefundos.s3-sa-east-1.amazonaws.com/cvm/cdi'
    )

    cursor = cnxn.cursor()
    # cursor.execute("Truncate Table [Fundos_05052020].[indicesDiarios]")
    # cursor.execute("Truncate Table [Fundos_05052020].[rentabilidadeMensal]")
    # cursor.execute("Truncate Table [Fundos_05052020].[rentabilidadeAnual]")

    if '<Error>' not in pageContent.text and pageContent.text!='':
        respJSON = pageContent.json()
        cotacoesMensais = []
        cotacoesAnuais = []
        if len(respJSON)>1:
            cotacaoAnterior = respJSON[0]
            i = 1
            for cotacao in respJSON[1:]:
                if cotacao['c'] > 0:
                    if cotacaoAnterior['c'] == 0:
                        cotacaoAnterior = cotacao
                    dataCotacaoAnterior = datetime.datetime.strptime(str(cotacaoAnterior['d']), '%Y%m%d')
                    dataCotacao = datetime.datetime.strptime(str(cotacao['d']), '%Y%m%d')
                    if dataCotacaoAnterior.month!=dataCotacao.month:
                        cotacoesMensais.append(cotacaoAnterior)
                    if dataCotacaoAnterior.year!=dataCotacao.year:
                        cotacoesAnuais.append(cotacaoAnterior)

                    cursorRent = cnxn.cursor()
                    cursorRent.execute("""SELECT [Ind_Data]
                          FROM [dbo].[indicesDiariosCDI] Where Ind_Data=?""", datetime.datetime.strptime(str(cotacao['d']), '%Y%m%d'))
                    if cursorRent.fetchone() is not None:
                        continue
                    sql = """INSERT INTO [dbo].[indicesDiariosCDI]
                           ([Ind_Data]
                           ,[Ind_Rent])
                        VALUES (?, ?)"""
                    cursor.execute(sql, datetime.datetime.strptime(str(cotacao['d']), '%Y%m%d'), 
                                  cotacao['c'])
                    cotacaoAnterior = cotacao
                    i = i + 1
        # print(cotacoesMensais)
        if len(cotacoesMensais)>1:
            cotacaoAnterior = cotacoesMensais[0]
            for cotacao in cotacoesMensais[1:]:
                #print(cotacao['d'])
                #print("{:.2%}".format(cotacao['c']/cotacaoAnterior['c']-1))
                cursorRent = cnxn.cursor()
                cursorRent.execute("""SELECT [Ind_Data]
                      FROM [dbo].[RentabilidadeMensalCDI] Where Ind_Data=DATEADD(month, DATEDIFF(month, 0, ?), 0)""", datetime.datetime.strptime(str(cotacao['d']), '%Y%m%d'))
                if cursorRent.fetchone() is not None:
                    continue
                sql = """INSERT INTO [dbo].[RentabilidadeMensalCDI]
                       ([Ind_Data]
                       ,[Ind_Rent])
                    VALUES (DATEADD(month, DATEDIFF(month, 0, ?), 0), ?)"""
                cursor.execute(sql, datetime.datetime.strptime(str(cotacao['d']), '%Y%m%d'), 
                              cotacao['c']/cotacaoAnterior['c']-1)
                cotacaoAnterior = cotacao
        # print(cotacoesAnuais)
        if len(cotacoesAnuais)>1:
            cotacaoAnterior = cotacoesAnuais[0]
            for cotacao in cotacoesAnuais[1:]:
                # print(cotacao['d'])
                # print("{:.2%}".format(cotacao['c']/cotacaoAnterior['c']-1))
                cursorRent = cnxn.cursor()
                cursorRent.execute("""SELECT [Ind_Data]
                      FROM [dbo].[RentabilidadeAnualCDI] Where Ind_Data=DATEADD(year, DATEDIFF(year, 0, ?), 0)""", datetime.datetime.strptime(str(cotacao['d']), '%Y%m%d'))
                if cursorRent.fetchone() is not None:
                    continue
                sql = """INSERT INTO [dbo].[RentabilidadeAnualCDI]
                       ([Ind_Data]
                       ,[Ind_Rent])
                    VALUES (DATEADD(year, DATEDIFF(year, 0, ?), 0), ?)"""
                cursor.execute(sql, datetime.datetime.strptime(str(cotacao['d']), '%Y%m%d'), 
                              cotacao['c']/cotacaoAnterior['c']-1)
                cotacaoAnterior = cotacao
    else:
        print('Erro nos dados.')
    # cursor.commit()
    cursor.close()


# In[16]:


cnxn = pyodbc.connect("Driver={SQL Server Native Client 11.0};"
                      "Server=localhost\\NOVAINSTANCIA2;"
                      "Database=Fundos_TCC;"
                      "Trusted_Connection=yes;")
cnxn.autocommit = True

cursor = cnxn.cursor()
cursor.execute("Truncate Table [dbo].[indicesDiariosCDI]")
cursor.execute("Truncate Table [dbo].[RentabilidadeMensalCDI]")
cursor.execute("Truncate Table [dbo].RentabilidadeAnualCDI")
carregaRentabilidadesCDI()
cnxn.close()


# In[ ]:





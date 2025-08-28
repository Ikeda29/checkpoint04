import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import statsmodels.api as sm

st.set_page_config(page_title="Analise", layout="wide")

st.header("Segurança no Estado de SP")
st.write("Dados utilizados do Governo do Estado São Paulo dos anos 2002 até 2021")
st.write("Variaveis:")
st.write("Ano: Quantitativa Continua")
st.write("Região: Qualitativa Nominal")
st.write("Homicidios Dolosos: Quantitativo Discreto")
st.write("Latrocinio: Quantitativo Discreto")
st.write("Estupros: Quantitativo Discreto")
st.write("Roubo de Veiculo: Quantitativo Discreto")
st.write("Roubo de Outros: Quantitativo Discreto")
dfgeral = pd.read_csv('./br_sp_gov_ssp_ocorrencias_registradas.csv')
df = dfgeral[['ano', 'regiao_ssp', 'homicidio_doloso', 'latrocinio', 'total_de_estupro', 'total_de_roubo_outros', 'roubo_de_veiculo']]
st.write(df.head())

st.subheader("Quais cidades possuem os maiores indices?")

crime_opcoes = {
    'Homicídio Doloso': 'homicidio_doloso',
    'Latrocínio': 'latrocinio',
    'Estupro': 'total_de_estupro',
    'Roubo de Veículo': 'roubo_de_veiculo',
    'Outros Roubos': 'total_de_roubo_outros'
}

crime_escolhido = st.selectbox("",list(crime_opcoes.keys()))

coluna_crime = crime_opcoes[crime_escolhido]
df_crime = df.groupby('regiao_ssp')[coluna_crime].sum().reset_index()
df_crime = df_crime.sort_values(by=[coluna_crime], ascending=False)

st.write(f"Total de {crime_escolhido} por Região")
st.dataframe(df_crime)

st.write("Analisando, podemos concluir que a região da Grande São Paulo, principalmente a Capital, sofrem com a criminalidade.")
st.write("Por outro lado, Presidente Prudente e Araçatuba possuem indices mais baixos de criminalidade.")

st.subheader("Total por ano")
df_agrupado = df.groupby('ano')[[
    'homicidio_doloso',
    'latrocinio',
    'total_de_estupro',
    'total_de_roubo_outros',
    'roubo_de_veiculo'
]].sum().reset_index()


st.dataframe(df_agrupado)

st.write("Analisando ano a ano, é possivel observar que em relação a 2002, a maioria dos crimes reduzio em 2021. Com excessão do estupro que apresentou crescimento desde 2009.")

colunas_crimes = df_agrupado.columns.drop('ano')
df_agrupado[colunas_crimes] = df_agrupado[colunas_crimes] / 365

st.subheader("Média de Crime por dia")
st.dataframe(df_agrupado)

st.write("Analisando a média de crimes, é possivel compreender melhor os dados, como de em 2002 ocorrerem 32 homicidios em média por dia. Em 2013 ")

st.subheader("Homicidio Doloso ao longo dos anos por Região")

regioes_disponiveis = df['regiao_ssp'].unique()
regiao_escolhida = st.selectbox("", regioes_disponiveis)

df_regiao = df[df['regiao_ssp'] == regiao_escolhida]

df_ano = df_regiao.groupby('ano')['homicidio_doloso'].sum().reset_index()

st.write(f"Homicídios Dolosos na Região {regiao_escolhida}")
st.dataframe(df_ano)

st.write("Podemos observar de que a taxa de homicidios dolosos na Capital e na Grande São Paulo tiveram quedas significativas.")
st.write("Campinas e Santos apresentaram pequena queda na taxa de homicidios dolosos.")
st.write("Enquanto as demais regiões mantem uma certa estabilidade.")

homicidio_por_regiao_ano = dfgeral.groupby(['ano', 'regiao_ssp'])['homicidio_doloso'].sum().reset_index()

fig = plt.figure(figsize=(15, 8))

for regiao in homicidio_por_regiao_ano['regiao_ssp'].unique():
    df_regiao = homicidio_por_regiao_ano[homicidio_por_regiao_ano['regiao_ssp'] == regiao]
    plt.plot(df_regiao['ano'], df_regiao['homicidio_doloso'], label=regiao)

plt.xlabel('Ano')
plt.ylabel('Número de Homicídios Dolosos')
plt.title('Homicídios Dolosos por Ano e Região SSP')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.grid(True)
st.pyplot(fig)

st.write("Observando o gráfico, podemos observar uma tendencia das taxa de homicidio doloso na Grande São Paulo e Capital cair.")

regions_of_interest = ['Capital', 'Grande São Paulo (exclui a Capital)']
df_filtered_regions = dfgeral[dfgeral['regiao_ssp'].isin(regions_of_interest)].copy()

df_capital = df_filtered_regions[df_filtered_regions['regiao_ssp'] == 'Capital'].copy()

homicidio_capital_por_ano = df_capital.groupby('ano')['homicidio_doloso'].sum().reset_index()

X_capital = homicidio_capital_por_ano[['ano']]
y_capital = homicidio_capital_por_ano['homicidio_doloso']
X_capital = sm.add_constant(X_capital)
model_capital = sm.OLS(y_capital, X_capital)
results_capital = model_capital.fit()

coefficient_capital = results_capital.params['ano']
standard_error_capital = results_capital.bse['ano']
critical_value = 1.96

lower_bound_capital = coefficient_capital - (critical_value * standard_error_capital)
upper_bound_capital = coefficient_capital + (critical_value * standard_error_capital)

df_grande_sp = df_filtered_regions[df_filtered_regions['regiao_ssp'] == 'Grande São Paulo (exclui a Capital)'].copy()

homicidio_grande_sp_por_ano = df_grande_sp.groupby('ano')['homicidio_doloso'].sum().reset_index()

X_grande_sp = homicidio_grande_sp_por_ano[['ano']]
y_grande_sp = homicidio_grande_sp_por_ano['homicidio_doloso']

X_grande_sp = sm.add_constant(X_grande_sp)

model_grande_sp = sm.OLS(y_grande_sp, X_grande_sp)
results_grande_sp = model_grande_sp.fit()

coefficient_grande_sp = results_grande_sp.params['ano']
standard_error_grande_sp = results_grande_sp.bse['ano']

critical_value = 1.96

lower_bound_grande_sp = coefficient_grande_sp - (critical_value * standard_error_grande_sp)
upper_bound_grande_sp = coefficient_grande_sp + (critical_value * standard_error_grande_sp)

st.write("Intervalo de Confiança de 95% para a taxa de mudança dos homicidios dolosos na Grande São Paulo:")
st.write(f"- 'Capital': ({lower_bound_capital:.4f}, {upper_bound_capital:.4f})")
st.write(f"- 'Grande São Paulo (exclui a Capital)': ({lower_bound_grande_sp:.4f}, {upper_bound_grande_sp:.4f})")

st.write("\nInterpretação dos Intervalos de Confiança:")
st.write("Esses intervalos de confiança representam o alcance, com 95% de certeza, no qual a media anual muda no numero de homicidios em cada região durante o periodo observado.")
st.write("No caso da Capital, a media anual cai entre 121.84 e 221.73 homicidios.")
st.write("Enquanto para a Grande São Paulo (excluindo a Capital), tem uma queda media entre 89.85 e 145.25 homicidios.")
st.write("\nComo nenhum dos intervalos inclui zero e ambos são abaixo de zero,podemos concluir com 95% de confiança de que a taxa de homicidios dolosos na Grande São Paulo e Capital irá diminuir nos próximos anos.")

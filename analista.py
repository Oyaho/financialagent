import os
import warnings
import csv
from textwrap import dedent
from dotenv import load_dotenv
from datetime import date
from pydantic import BaseModel, Field
from typing import List, Dict
import time

# --- IMPORTS CORRIGIDOS E ESSENCIAIS ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate, BasePromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage


# --- 1. DEFINIÇÃO DO MODELO DE RELATÓRIO FINANCEIRO (Pydantic) ---
class AnaliseAcionaria(BaseModel):
    """Estrutura do Relatório de Análise de Ações."""
    data_relatorio: str = Field(description="Data de geração do relatório (formato YYYY-MM-DD).")
    empresa: str = Field(description="Nome da empresa e Ticker (ex: Netflix - NFLX).")
    valor_atual_acao: str = Field(description="O preço de negociação atual da ação.")
    sumario_executivo: str = Field(description="Resumo conciso das principais conclusões de investimento (máximo 4 frases).")
    noticias_relevantes: List[str] = Field(description="Lista das 3 principais notícias de mercado e seu impacto, resumidas em uma frase cada.")
    analise_financeira_resumida: str = Field(description="Resumo dos dados financeiros importantes (receita, lucro, FCF) extraídos do documento oficial. Se não houver dados de documento, use 'N/A - Foco em Notícias'.")
    sentimento_geral: str = Field(description="Sentimento do mercado baseado em fatos (Ex: Positivo/Negativo/Misto) e breve justificativa.")
    recomendacao_simplificada: str = Field(description="Recomendação simples de investimento (Ex: Manter, Comprar, Vender) baseada na análise.")


# --- 2. FERRAMENTA DE ANÁLISE DE DOCUMENTOS (RAG SIMULADO) ---
@tool
def consult_document_rag(report_url: str) -> str:
    """
    Use esta ferramenta para ler e resumir dados financeiros importantes 
    (Receita, Lucro Líquido, FCF) do relatório financeiro oficial (URL fornecida).
    Retorna um resumo conciso dos números principais para o Agente Pesquisador.
    """
    print(f"\n[RAG TOOL INVOCADA] -> Analisando URL: {report_url[:50]}...")
    
    if not report_url or 'N/A' in report_url or 'Sem URL' in report_url:
        return "Nenhum URL de relatório financeiro válido foi fornecido. Dados fundamentalistas indisponíveis."
    
    # --- SIMULAÇÃO: Retorna dados fixos ou baseados no ticker ---
    if 'NFLX' in report_url:
        return dedent("""
        Análise RAG do 10-K (2023): A Receita totalizou $33.7 bilhões, um aumento de 6.7% A/A.
        O Lucro Líquido foi de $5.4 bilhões. O Fluxo de Caixa Livre (FCF) foi robusto em $6.9 bilhões,
        indicando forte geração de caixa e saúde financeira.
        """)
    elif 'TSLA' in report_url:
        return dedent("""
        Análise RAG do 10-K (2023): A Receita atingiu $96.8 bilhões. Lucro Líquido: $15.0 bilhões.
        O FCF foi de $4.4 bilhões. O relatório destaca margens de lucro sob pressão devido a cortes de preços.
        """)
    else:
        return dedent("""
        Análise RAG (Simulada): Receita de $150 bilhões no último ano. Lucro líquido de $40 bilhões. 
        A empresa reportou forte recompra de ações e foca em serviços para crescimento futuro.
        """)


# --- 3. FUNÇÃO DE LEITURA DA LISTA (CSV) ---
def ler_lista_empresas_csv(caminho_arquivo="dados_empresas.csv") -> List[Dict[str, str]]:
    """Lê a lista de empresas e URLs de relatório do arquivo CSV."""
    dados = []
    try:
        with open(caminho_arquivo, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                dados.append(row)
        print(f"✅ Lista de empresas e URLs carregada ({len(dados)} itens).")
        return dados
    except FileNotFoundError:
        print(f"❌ ERRO: Arquivo '{caminho_arquivo}' não encontrado. Criando um exemplo.")
        exemplo_data = [
            {"Empresa": "Netflix", "Ticker": "NFLX", "Relatorio_URL": "N/A - Sem URL"},
            {"Empresa": "Tesla", "Ticker": "TSLA", "Relatorio_URL": "N/A - Sem URL"}
        ]
        with open(caminho_arquivo, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['Empresa', 'Ticker', 'Relatorio_URL']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in exemplo_data:
                writer.writerow(row)
        return exemplo_data


# --- 4. CONFIGURAÇÃO LOCAL E LLM ---
load_dotenv()
if "LANGCHAIN_TRACING_V2" in os.environ: del os.environ["LANGCHAIN_TRACING_V2"]
if "LANGCHAIN_API_KEY" in os.environ: del os.environ["LANGCHAIN_API_KEY"]
warnings.filterwarnings("ignore", category=UserWarning)

print("Ambiente e chaves carregadas.")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0) 
tavily_tool = TavilySearchResults(max_results=3) 
tools = [tavily_tool, consult_document_rag] 


# --- 5. DEFINIÇÃO DO AGENTE PESQUISADOR (Agent 1 - ReAct) ---
# REESCRITA DO PROMPT PARA GARANTIR QUE TODAS AS VARIÁVEIS SEJAM RECONHECIDAS
REACT_PROMPT_TEMPLATE = """
Você é um agente útil, proficiente em análise financeira, focado em busca de fatos.
Sua única tarefa é usar as ferramentas disponíveis para coletar as informações solicitadas.
Responda à questão o melhor que puder. Você tem acesso às seguintes ferramentas:

{tools}

Use o seguinte formato de raciocínio, mantendo a ordem:

Question: a questão que você precisa responder
Thought: você deve sempre pensar sobre o que fazer, qual ferramenta usar e qual informação buscar.
Action: a ação a ser tomada, sempre uma das [{tool_names}]
Action Input: o input para a ação (sem aspas)
Observation: o resultado da ação
... (este Thought/Action/Action Input/Observation pode se repetir N vezes)
Thought: Eu sei a resposta final e detalhada
Final Answer: a resposta final e detalhada para a questão original (incluindo fatos da web e do documento)

Begin!

Question: {input}
Thought: {agent_scratchpad}
"""
# A função create_react_agent preenche {tools}, {tool_names}, e {agent_scratchpad} automaticamente
prompt_researcher: BasePromptTemplate = PromptTemplate.from_template(REACT_PROMPT_TEMPLATE) 

agent_researcher_model = create_react_agent(llm, tools, prompt_researcher)

# MUDANÇA CRÍTICA: Desativar a verbosidade do raciocínio ReAct
agent_executor_researcher = AgentExecutor(
    agent=agent_researcher_model, 
    tools=tools, 
    verbose=False, # VERBOSIDADE DESATIVADA PARA LIMPEZA DO CONSOLE
    handle_parsing_errors=True
)

# --- 6. AGENTE RELATOR (Agent 2 - Geração de JSON) ---
relator_prompt_template = """
Você é um Analista de Investimentos Sênior. Sua tarefa é compilar os FATOS DA PESQUISA
fornecidos sobre a empresa {empresa} em um relatório profissional.

Siga rigorosamente as instruções de formatação e preencha todos os campos do objeto JSON.
Use a data de hoje para o campo 'data_relatorio'.
Se não houver dados fundamentalistas, preencha o campo 'analise_financeira_resumida' com 'N/A - Foco em Notícias'.
Não adicione texto extra, apenas o objeto JSON.

FATOS DA PESQUISA (Contém dados da Web e do Documento):
{fatos_consolidados}
"""
relator_prompt = PromptTemplate.from_template(relator_prompt_template)
relator_chain = relator_prompt | llm.with_structured_output(AnaliseAcionaria)


# --- 7. LOOP DE EXECUÇÃO PRINCIPAL ---
empresas_para_analisar = ler_lista_empresas_csv()

for dados_empresa in empresas_para_analisar:
    empresa_nome_completo = f"{dados_empresa['Empresa']} ({dados_empresa['Ticker']})"
    relatorio_url = dados_empresa['Relatorio_URL']

    # Banners e inícios de processo mais concisos
    print(f"\n=======================================================")
    print(f"| 📊 INICIANDO ANÁLISE: {empresa_nome_completo}")
    print(f"=======================================================")

    pergunta_pesquisa = f"""
    Para a empresa {empresa_nome_completo}, gere uma análise consolidada.
    1. Use a ferramenta 'tavily_search' para encontrar o VALOR ATUAL da ação e as três notícias mais recentes.
    2. Use a ferramenta 'consult_document_rag' com o input '{relatorio_url}' para extrair o resumo financeiro dos documentos.
    3. Consolide TODAS as informações (valor da ação, notícias da web e fatos do documento) para a resposta final.
    """

    # 7.2. EXECUÇÃO DA CADEIA DE PESQUISA
    try:
        research_result = agent_executor_researcher.invoke({"input": pergunta_pesquisa})
        fatos_consolidados = research_result['output']
    except Exception as e:
        fatos_consolidados = f"Erro na pesquisa web para {empresa_nome_completo}: {e}"
        print(f"❌ [ERRO NA PESQUISA] Falha na coleta de dados para {empresa_nome_completo}.")
        continue 
    
    # 7.3. GERAÇÃO DO RELATÓRIO E SALVAMENTO
    try:
        relatorio_objeto: AnaliseAcionaria = relator_chain.invoke({
            "empresa": empresa_nome_completo,
            "fatos_consolidados": fatos_consolidados
        })
        
        # --- GERAÇÃO DO ARQUIVO MARKDOWN ---
        file_name_clean = empresa_nome_completo.replace(' ', '_').replace('(', '').replace(')', '')
        
        final_report_markdown = dedent(f"""
# 📄 Relatório de Análise Acionária: {relatorio_objeto.empresa}
**Data da Análise:** {date.today()}

---

## I. Sumário Executivo
{relatorio_objeto.sumario_executivo}

| Indicador | Detalhe |
| :--- | :--- |
| **Valor Atual da Ação** | **{relatorio_objeto.valor_atual_acao}** |
| **Recomendação** | **{relatorio_objeto.recomendacao_simplificada.upper()}** |
| **Sentimento Geral** | {relatorio_objeto.sentimento_geral} |

---

## II. Contexto e Notícias Relevantes
### Destaques das Últimas Notícias
* {"\n* ".join(relatorio_objeto.noticias_relevantes)}

---

## III. Análise Fundamentalista (Resumo de Documentos Oficiais - RAG)
{relatorio_objeto.analise_financeira_resumida}

---

**Nota:** A análise fundamentalista foi extraída da URL fornecida e processada pela ferramenta RAG (atualmente em modo simulação).
""")
        # Salvar como arquivo Markdown
        with open(f"Relatorio_{file_name_clean}.md", "w", encoding="utf-8") as f:
            f.write(final_report_markdown)

        # Print de sucesso
        print(f"✅ RELATÓRIO CONCLUÍDO! Salvo como 'Relatorio_{file_name_clean}.md'")

    except Exception as e:
        print(f"❌ [ERRO NA GERAÇÃO] Falha na estruturação JSON do relatório para {empresa_nome_completo}: {e}")
    
    # DELAY
    DELAY_SECONDS = 7 
    print(f"[DELAY] ⏱️ Aguardando {DELAY_SECONDS} segundos para evitar exceder a quota da API Gemini...")
    time.sleep(DELAY_SECONDS)

print("\n=======================================================")
print("| ✨ PROCESSAMENTO DE TODAS AS EMPRESAS CONCLUÍDO! ✨ |")
print("=======================================================")
# 🤖 Financial Analyst Agent (FAA) - Análise de Ações com Gemini e LangChain

## 🎯 Descrição do Projeto

Este projeto implementa um **Agente de Análise Financeira (FAA)** utilizando o poder de raciocínio de Large Language Models (LLMs) do **Google Gemini (modelo gemini-2.5-flash)** orquestrado pelo framework **LangChain (versão 1.0.7+)**.

O Agente é configurado como um pipeline de duas fases:
1.  **Agente Pesquisador (ReAct):** Utiliza ferramentas de busca em tempo real (Web Search via Tavily) para coletar as últimas notícias, o contexto de mercado e os fatos relevantes sobre uma ação-alvo.
2.  **Agente Relator:** Consolida as informações factuais coletadas e as sintetiza em um **Relatório de Análise de Ações** estruturado em Markdown, contendo Contexto Atual, Análise de Sentimento e um Sumário Executivo.

O projeto é **100% gratuito** e segue a arquitetura modular moderna do LangChain.

## ⚙️ Requisitos e Instalação

### 1. Ambiente

É altamente recomendável utilizar um ambiente virtual (`venv`) para isolar as dependências do projeto.

```bash
# Crie e ative o ambiente virtual
python -m venv venv

# No Windows/PowerShell:
.\venv\Scripts\activate
# No Linux/macOS:
source venv/bin/activate

# Instalar requirements
pip install -r requirements.txt

# Adicione as chaves ao arquivo .env neste formato:

# Chave da Google (LLM Gemini)
GOOGLE_API_KEY="SUA_CHAVE_AQUI"

# Chave da Tavily (Web Search Tool)
TAVILY_API_KEY="SUA_CHAVE_AQUI"

# Após configurar o ambiente e o arquivo .env, execute o agente através do script principal:
python analista_agent.py


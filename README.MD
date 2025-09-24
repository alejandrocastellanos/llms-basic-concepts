# LLMs Basic Concepts & LangChain Playground

This repository contains a collection of Python scripts and mini-projects focused on learning and experimenting with Large Language Models (LLMs), LangChain, and OpenAI APIs. It is designed as a hands-on course or playground for understanding core LLM concepts, prompt engineering, chaining, memory, and retrieval-augmented generation.

## Table of Contents
- [Project Structure](#project-structure)
- [Key Features](#key-features)
- [Dependencies](#dependencies)
- [How to Use](#how-to-use)
- [AgentR5: RAG Example](#agentr5-rag-example)
- [License](#license)

## Project Structure

```
llms-basic-concepts/
├── AgentR5/
│   ├── agent_r5.py              # RAG chatbot for Grupo R5 (fintech)
│   ├── build_r5_vectorstore.py  # Build FAISS vectorstore for R5
│   └── faiss_r5/                # Saved FAISS vectorstore
├── basic.py                     # Simple QA with LangChain LLMChain
├── basic_concepts.py            # Prompt template and chain example
├── chat_with_memory.py          # ConversationChain with memory
├── compose_chaina.py            # SequentialChain: summarize, translate, hashtag
├── open-ai-web.py               # Viral news agent using OpenAI & Tavily
├── sequential_chain.py           # Multi-step chain with explanations
├── technical_assistant.py        # Technical assistant for software topics
├── requirements.txt             # Python dependencies
├── .env                         # Environment variables (add your OpenAI key)
├── LICENSE
└── README.md
```

## Key Features
- **Prompt Engineering**: Learn to design and use prompt templates for LLMs.
- **LangChain Chains**: Examples of LLMChain, SequentialChain, and ConversationChain.
- **Memory**: Maintain conversational context with buffer memory.
- **Retrieval-Augmented Generation (RAG)**: Use FAISS vectorstore for domain-specific QA (Grupo R5 example).
- **Agents**: Custom agent for viral news retrieval with system prompt control.
- **Technical Assistant**: Specialized assistant for software architecture and engineering topics.

## Dependencies
Install all dependencies with:
```bash
pip install -r requirements.txt
```
Key packages:
- `langchain`, `langchain-openai`, `langchain-community`
- `openai`
- `python-dotenv`
- `faiss` (via langchain)
- `tavily-search` (for viral news agent)

## How to Use
1. **Set up your environment**:
   - Copy `.env.example` to `.env` (if available) and add your OpenAI API key:
     ```
     OPENAI_API_KEY=sk-...
     ```
2. **Run scripts**:
   - Try any script directly:
     ```bash
     python basic.py
     python compose_chaina.py
     python technical_assistant.py
     python open-ai-web.py
     # etc.
     ```
3. **AgentR5 (RAG Example)**:
   - Build the vectorstore (run once):
     ```bash
     python AgentR5/build_r5_vectorstore.py
     ```
   - Start the R5 assistant:
     ```bash
     python AgentR5/agent_r5.py
     ```

## AgentR5: RAG Example
- **Purpose**: Answer questions about Grupo R5 using a local FAISS vectorstore built from structured data.
- **How it works**: Embeds knowledge about Grupo R5; uses retrieval to ground LLM answers.
- **Try it**: Run the scripts above and ask questions about Grupo R5's services, platform, or reputation.

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

*For more details, see the code and comments in each script. This repo is a practical playground for LLM and LangChain experimentation.*

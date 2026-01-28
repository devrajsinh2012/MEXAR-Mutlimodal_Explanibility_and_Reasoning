# MEXAR: Multimodal Explainability and Reasoning Assistant 🧠

> **Final Year Major Project**
> *An Advanced Agentic AI System for Transparent, Multimodal Reasoning*

---

## 🌟 Project Overview

**MEXAR** is a cutting-edge Artificial Intelligence framework designed to bridge the gap between black-box deep learning models and human interpretability. As explainability becomes paramount in AI adoption for critical sectors like healthcare, finance, and law, MEXAR provides a robust solution that not only generates answers but **explains its reasoning process**.

This project represents a comprehensive "Ultimate" system that integrates **Multimodal RAG (Retrieval-Augmented Generation)**, **Knowledge Graphs**, and **Causal Reasoning** to deliver precise, citation-backed answers from diverse data sources including text documents, images, and audio.

### 🎯 Core Objectives
- **Multimodal Understanding**: Seamlessly process and correlate information from PDFs, images (Vision), and audio (Whisper).
- **Explainability First**: Every answer comes with *Source Attribution* (citations) and a *Faithfulness Score*.
- **Domain Guardrails**: Strict boundary enforcement to prevent hallucinations and ensure domain-specific relevance.
- **Interactive Experience**: A premium, React-based UI featuring real-time reasoning visualization and voice interaction.

---

## 🏗️ System Architecture & Technologies

The project is built on a modern, scalable Microservices-based architecture:

| Component | Tech Stack | Role |
|-----------|------------|------|
| **Frontend** | React 18, Material UI, Recharts | Interactive Dashboard & Chat Interface |
| **Backend** | FastAPI, Python 3.9+ | High-performance Async REST API |
| **Database** | Supabase (PostgreSQL + pgvector) | Vector Store & Relational Data Persistence |
| **AI Engine** | Groq (LPU), FastEmbed | Ultra-fast LLM Inference & Embedding Generation |
| **RAG Pipeline**| Hybrid Search (Semantic + Keyword) | Advanced Retrieval with RRF Fusion |
| **Multimodal** | OpenAI Whisper, Vision Models | Audio transcription & Image Analysis |

---

## 📂 Detailed Repository Structure

This repository contains the complete source code for the MEXAR ecosystem. Below is an in-depth breakdown of the project organization:

### 1. `MEXAR_Ultimate/`
The core directory containing the producution-ready application.

#### **Backend (`/backend`)**
The brain of the system, powered by **FastAPI**.
- **`api/`**: Contains all REST API route definitions.
    - `auth.py`: Handles JWT-based user authentication and session management.
    - `chat.py`: The core chat endpoint processing multimodal inputs and streaming responses.
    - `agents.py`: Manages the lifecycle of domain-specific AI agents.
    - `compile.py`: Handles the "Knowledge Compilation" process (vectorization of datasets).
- **`core/`**: Essential configuration and reliable singletons.
    - `config.py`: Centralized environment variable management.
    - `database.py`: SQLAlchemy database connection and session handling.
- **`modules/`**: The specialized AI logic units.
    - `knowledge_compiler.py`: Chunks, embeds, and indexes document data into pgvector.
    - `reasoning_engine.py`: Orchestrates the RAG pipeline, retrieving and ranking context.
    - `faithfulness.py`: Algorithms to score how well an answer is supported by facts.
    - `prompt_analyzer.py`: Validates user queries against domain scope.
    - `data_validator.py`: Robust file checking for uploaded datasets (PDF, JSON, CSV, etc).
- **`services/`**: Business logic layer connecting API routes to core modules.
    - `agent_service.py`: Logic for creating and configuring agents.
    - `chat_service.py`: Manages conversation state and context windowing.
- **`models/`**: SQL Database schemas (ORM).
    - `agent.py`, `user.py`, `chunk.py`: Defines the data structure for persistence.

#### **Frontend (`/frontend`)**
A modern, responsive Single Page Application (SPA) built with **React**.
- **`src/pages/`**: The main application views.
    - `Landing.jsx`: The hero entry page showcasing project capabilities.
    - `Dashboard.jsx`: User's central hub for managing agents and files.
    - `Chat.jsx`: The complex chat interface supporting streamed text, markdown rendering, and file uploads.
    - `AgentCreation.jsx`: Wizard-style interface for building new AI agents.
- **`src/components/`**: Reusable UI widgets.
    - `Sidebar.jsx`, `Navbar.jsx`: Navigation elements.
    - `FileUpload.jsx`: Drag-and-drop zone with validation.
    - `ThinkingAnimation.jsx`: Visual feedback for the AI's reasoning state.
- **`src/contexts/`**: Global state management (User Auth, Theme).

### 2. Other Directories
- **`MEXAR_RAG/`**: Experimental folder containing isolated RAG implementation tests and benchmarks.
- **`MEXAR_Nano/`** & **`MEXAR_Lite/`**: Lightweight prototypes developed during the early research phases of the project.
- **`Document/`**: Contains project documentation, thesis drafts, and diagrams.
- **`daataset/`**: A collection of sample datasets (Medical, Financial, Legal) used for testing the system's adaptability.

---

## ✨ Key Features Showcase

### 🔍 Advanced RAG Pipeline
Unlike simple vector search, MEXAR employs a **Hybrid Search** mechanism. It combines:
1.  **Dense Retrieval**: Using vector embeddings to find semantic meaning.
2.  **Sparse Retrieval**: Keyword-based search (BM25) for exact definitions.
3.  **Reranking**: A Cross-Encoder model re-scores top results for maximum precision.

### 🛡️ Trust & Safety
- **Attribution**: The system cites its sources (e.g., `[Doc 1, Page 5]`), allowing users to verify claims.
- **Faithfulness Score**: A self-evaluation metric (0-100%) indicating the model's confidence based *strictly* on retrieved data.

### 🗣️ Multimodal Interaction
- **Voice Mode**: Users can speak to MEXAR using Whisper integration, and it replies with high-quality TTS (ElevenLabs).
- **Vision**: Upload charts or diagrams, and MEXAR will analyze them as part of the reasoning context.

---

## 👨‍💻 Project Team

This Major Project is proudly presented by:

**Devrajsinh Gohil** & **Jay Nasit**

Under the expert guidance of:

**Prof. Om Prakash Suthar**

---

# CodeScribble-AI

Heurist Agent Framework - Final Submission

Overview

The Heurist Agent Framework is an advanced autonomous AI system designed to interact across multiple platforms, process multimodal inputs, and utilize state-of-the-art language models. This project integrates image generation, voice interaction, and efficient data retrieval using PostgreSQL with pgvector.

Features

Autonomous AI Agents: Performs intelligent tasks without manual intervention.

Multimodal Support: Handles text, voice, and image-based interactions.

Image Generation: Uses AI models for creating high-quality visuals.

Vector Database Integration: Stores and retrieves embeddings efficiently with pgvector.

Deployment on Social Media: Supports Telegram, Discord, and Twitter.

Installation & Setup

Prerequisites

Python <3.8

PostgreSQL (with pgvector extension)

Git

Virtual Environment (venv)

Steps to Install

Clone the Repository:

git clone https://github.com/Khushbu710/CodeScribble-AI
cd heurist-agent-framework-main

Set Up a Virtual Environment:

python -m venv venv
source venv/bin/activate   # On macOS/Linux
venv\Scripts\activate      # On Windows

Install Dependencies:

pip install -r requirements.txt

Database Configuration
VECTOR_DB_HOST=127.0.0.1
VECTOR_DB_PORT=5432
VECTOR_DB_NAME=heuristdb
VECTOR_DB_USER=heuristuser
VECTOR_DB_PASSWORD=khushbu.sharma7105
VECTOR_DB_TABLE=vector_table

Start PostgreSQL and Create a Database:

psql -U postgres
CREATE DATABASE heurist_db;

Enable pgvector Extension:

\c heurist_db
CREATE EXTENSION vector;

Set Up Environment Variables (.env file):

VECTOR_DB_HOST=localhost
VECTOR_DB_PORT=5432
VECTOR_DB_NAME=heurist_db
VECTOR_DB_USER=your_db_user
VECTOR_DB_PASSWORD=your_db_password
VECTOR_DB_TABLE=your_vector_table

Running the Project

Ensure PostgreSQL is running and configured.

Run the AI agent:

python main.py

The system will initialize and start processing interactions.

Deployment Guide

The project can be deployed on cloud platforms like AWS, Heroku, or Railway.

Configure the .env file correctly before deployment.

Make sure the PostgreSQL database is accessible from the deployment environment.

Contribution

Contributions are welcome! Feel free to fork the repository, work on improvements, and submit a pull request.

License

This project is licensed under the MIT License.

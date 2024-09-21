This project is an AI-powered competitive analysis tool designed for startups. It leverages a LangGraph agent to guide users in refining the scope of their competitive analysis. The tool employs a robust Retrieval-Augmented Generation (RAG) pipeline, incorporating multiple advanced retrieval models such as ColBERT, SPLADE, DPR, BM25-BERT Hybrid, ANCE, MiniLM, and Reformer. This enables the extraction of relevant insights from various online sources like news articles, market reports, blogs, research papers, and more. 

A user-friendly Streamlit-based dashboard provides a seamless interactive experience, allowing users to input their industry, competitors, and key focus areas. The tool then retrieves and presents competitive data through insightful visualizations. Users have the flexibility to iteratively refine their searches and compare the performance of different retrieval models in real-time.

To get started, follow these installation and execution instructions.

Installation:
1. Ensure you have Python (3.x) installed on your Windows 10 machine.
2. Create a virtual environment to manage project dependencies:
   - python -m venv .venv
3. Activate the virtual environment:
   - .venv\Scripts\activate
4. Install the required packages listed in the 'requirements.txt' file:
   - pip install -r requirements.txt
5. Obtain and set your API keys:
   - Get an OpenAI API key and set it as an environment variable named 'OPENAI_API_KEY'. You can do this by adding the following line to your terminal's configuration file (e.g., .bashrc or .zshrc): 
     - export OPENAI_API_KEY=<YOUR_API_KEY>
   - Get a Hugging Face API token and set it as an environment variable named 'HUGGINGFACEHUB_API_TOKEN': 
     - export HUGGINGFACEHUB_API_TOKEN=<YOUR_API_TOKEN>
   - Replace <YOUR_API_KEY> and <YOUR_API_TOKEN> with your actual keys.

Running the Application:

1. Navigate to the 'streamlit_app' directory:
   - cd streamlit_app
2. Execute the Streamlit application:
   - streamlit run dashboard.py
3. Access the dashboard in your web browser at the address shown in the terminal (typically http://localhost:8501).

Directory Structure:

The project is structured as follows:

📦 AI_Competitive_Analysis_Tool/
├── 📁 app/
│   ├── 📁 models/
│   │   ├── colbert_model.py                # ColBERT retrieval model
│   │   ├── splade_model.py                 # SPLADE retrieval model
│   │   ├── dpr_model.py                    # DPR retrieval model
│   │   ├── bm25_bert_model.py             # BM25-BERT Hybrid model
│   │   ├── ance_model.py                   # ANCE model
│   │   ├── minilm_model.py                 # MiniLM model
│   │   └── reformer_model.py              # Reformer model
│   ├── 📁 rag_pipeline/
│   │   ├── iterative_rag.py                # Iterative RAG pipeline
│   │   └── retriever.py                    # Base retrieval logic
│   ├── 📁 streamlit_app/
│   │   ├── dashboard.py                    # Streamlit UI/UX
│   │   └── visualizations.py               # Data visualizations
│   ├── 📁 langgraph_agent/
│   │   └── langgraph_chatbot.py            # LangGraph agent
│   ├── 📁 data_acquisition/
│   │   ├── company_website_scraper.py      # Website data scraper
│   │   ├── social_media_analyzer.py        # Social media analysis
│   │   └── financial_data_extractor.py     # Financial data extraction
│   ├── 📁 data/
│   │   ├── sample_competitive_data.csv     # Sample data
│   │   └── scraped_results.json            # Scraped results
│   └── __init__.py                         # App module initialization
├── 📁 config/
│   ├── model_config.json                   # Model configurations
│   └── app_config.json                     # Application settings
├── 📁 tests/
│   ├── test_colbert.py                     # ColBERT model tests
│   ├── test_splade.py                      # SPLADE model tests
│   ├── test_dpr.py                         # DPR model tests
│   ├── test_bm25_bert.py                  # BM25-BERT Hybrid model tests
│   ├── test_ance.py                        # ANCE model tests
│   ├── test_minilm.py                      # MiniLM model tests
│   ├── test_reformer.py                   # Reformer model tests
│   └── test_rag_pipeline.py                # RAG pipeline tests
├── 📁 logs/
│   ├── retrieval_log.txt                   # Retrieval operation logs
│   └── error_log.txt                       # Error logs
├── 📁 scripts/
│   └── exploratory_analysis.py             # Data exploration
├── 📁 docs/
│   └── user_guide.md                       # Tool usage documentation
├── requirements.txt                        # Python dependencies
├── setup.py                                # Project setup
└── README.md                               # Project overview

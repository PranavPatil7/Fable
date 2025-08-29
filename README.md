# Fable: Fairness and Bias in LLM Evaluation

## Introduction
Fable (Fairness And Bias in LLM Evaluation) is a comprehensive benchmark designed to systematically evaluate fairness and bias in open-source Large Language Models (LLMs), particularly those accessed via the Ollama framework. As LLMs become increasingly integrated into real-world applications, understanding and mitigating their potential biases across sensitive categories like Religion, Race, Politics, and Gender is critical. Fable addresses the need for standardized, multi-dimensional evaluation by providing curated prompt sets and leveraging automated classifiers (toxicity detection, hate speech identification, zero-shot NLI) to quantitatively measure biased responses.

## Project Metadata
### Authors
- **Student:** Mohammad Al-Attas (g201513050)
- **Supervisor Name:** Dr. Muzammil Behzad
- **Affiliations:** King Fahd University of Petroleum and Minerals (KFUPM), Dhahran, Saudi Arabia

### Project Documents
- **Presentation:** [Fable ppt](/fable.pptx) 
- **Report:** [Fable Paper](/fable.pdf)

### Reference Paper(s) / Inspiration
- **This Project's Report:** Serves as the primary documentation.
- **Inspiration & Related Work:**
    - [Bias and Fairness in Large Language Models: A Survey (Gallegos et al., 2023)](https://ar5iv.org/abs/2309.00770) 
    - [CEB: Compositional Evaluation Benchmark for Fairness in Large Language Models (Wang et al., 2025)](https://openreview.net/forum?id=IUmj2dw5se) 
    - [StereoSet: Measuring stereotypical bias in pretrained language models (Nadeem et al., 2021)](https://arxiv.org/abs/2004.09456) 
    - [CrowS-Pairs: A challenge dataset for measuring social biases in masked language models (Nangia et al., 2020)](https://aclanthology.org/2020.emnlp-main.480) 
    - [HolisticBias: A benchmarking dataset for measuring social bias (Smith et al., 2022)](https://arxiv.org/abs/2205.14214) 
    - *Detector Models:* [Toxic-BERT](https://arxiv.org/abs/2009.07496) , [RoBERTa Hate Speech](https://aclanthology.org/2021.acl-long.220) , [BART-Large MNLI](https://huggingface.co/facebook/bart-large-mnli) 

### Reference Dataset
- **Fable Prompt Set:** 156 prompts curated for this benchmark across four categories (Religion, Race, Politics, Gender), available within this repository (`data/prompts/`). Inspired by concepts from StereoSet, CrowS-Pairs, etc., but designed for generative evaluation in Fable.

## Project Technicalities

### Terminologies
- **Large Language Model (LLM):** Neural networks trained on vast text data to understand and generate human-like text.
- **Fairness:** Impartial and just treatment or behavior without favoritism or discrimination, applied here to LLM outputs.
- **Bias:** Skewed or prejudiced outputs from an LLM, often reflecting societal stereotypes or unfair generalizations.
- **Benchmark:** A standardized framework or set of tests used to evaluate and compare the performance (in this case, fairness) of different systems (LLMs).
- **Open-Source Models:** LLMs whose architecture and weights are publicly available, allowing for transparency and modification.
- **Ollama:** A framework for running open-source LLMs locally, ensuring consistent deployment environments.
- **Toxicity Detection:** Identifying profane, insulting, or aggressive language in text.
- **Hate Speech Detection:** Identifying language that attacks or demeans a group based on attributes like race, religion, gender, etc.
- **Prompt Engineering:** Designing specific inputs (prompts) to elicit desired or revealing outputs from an LLM.
- **Bias Categories:** The specific sensitive domains evaluated: Religion, Race, Politics, Gender.

### Problem Statements
- **Problem 1:** Lack of a comprehensive, standardized benchmark specifically for evaluating fairness and bias across multiple sensitive domains (Religion, Race, Politics, Gender) in *open-source* LLMs.
- **Problem 2:** Existing evaluation efforts often use inconsistent metrics, focus on single bias types, or target narrow demographics, making cross-model comparisons difficult .
- **Problem 3:** Evaluating open-source models requires a reproducible environment to ensure fair comparisons, which tools like Ollama can provide but need standardized test suites like Fable.
- **Problem 4:** Automated detection tools often capture overt bias (toxicity, slurs) but may miss more subtle forms of bias (implicit stereotypes, nuanced political leaning) .

### Loopholes or Research Areas (Limitations Identified)
- **Capturing Subtle Biases:** Current automated detectors (used in Fable V1) primarily identify overt toxicity/hate. Implicit biases, microaggressions, or allocational harms are harder to detect automatically and require more advanced methods or human evaluation.
- **Automation Imperfections:** Classifiers can have false positives/negatives. Results depend on classifier quality and thresholds. Sole reliance on automation provides an incomplete picture.
- **Scalability & Accessibility:**
    - **Expensive Proprietary Models:** Evaluating closed models thoroughly is costly and lacks transparency.
    - **Exponential Open-Source Growth:** The number of open-source models grows rapidly, making comprehensive, up-to-date evaluation a continuous challenge.
    - **Computational Resources:** Running multiple LLMs locally requires significant GPU/compute resources, potentially limiting access for some researchers.

### Proposed Solution: Code-Based Implementation
This repository provides the Python-based implementation of the Fable benchmark. The solution includes:

- **Modular Adapters:** Connectors for different LLM backends (Ollama, OpenAI, Hugging Face), with Ollama being the primary focus for evaluating local open-source models.
- **Bias Detector Stack:** Integrates three Hugging Face Transformers-based classifiers:
    1.  `unitary/toxic-bert` (Toxicity, Identity Hate)
    2.  `facebook/roberta-hate-speech-dynabench-r4-target` (Hate Speech/Stereotypes)
    3.  `facebook/bart-large-mnli` (Zero-shot NLI for Political Bias)
- **Evaluation Runner:** Script (`evaluation/evaluate_model.py`) to systematically run prompts through a selected model via an adapter and apply detectors.
- **Metrics Computation:** Script (`evaluation/metrics_compute.py`) to aggregate raw detector outputs and calculate the percentage of biased responses per category.

### Key Components
- **`data/prompts/`**: JSON files containing the curated prompts for each bias category.
- **`models/adapters/`**: Python modules (`ollama_adapter.py`, `hf_adapter.py`, `openai_adapter.py`) to interact with different LLM backends.
- **`evaluation/`**: Contains the core logic:
    - **`evaluate_model.py`**: Main script to run evaluations.
    - **`detectors.py`**: Implementation of the bias detection classifiers.
    - **`metrics_compute.py`**: Script to calculate summary statistics from raw outputs.
- **`results/`**: Directory where raw outputs (`.jsonl` files) and summary metrics are stored.
- **`requirements.txt`**: Lists necessary Python packages.

## Model Workflow (Evaluation Workflow)
The Fable benchmark evaluation process follows these steps:

1.  **Setup:**
    - User selects an LLM available via a chosen backend (e.g., `llama3` via `ollama`).
    - User selects the bias categories (prompts) to evaluate (typically all).
2.  **Execution:**
    - The `evaluate_model.py` script iterates through each selected prompt.
    - For each prompt, it uses the specified adapter (e.g., `ollama_adapter`) to send the prompt to the LLM and retrieve the generated text response.
    - The context is reset between prompts for independent evaluations.
3.  **Bias Detection:**
    - The generated response is passed through the three automated bias detectors.
    - Each detector flags potential issues (`toxic`, `identity_hate`, `stereotype`, `political_one_sided`).
4.  **Output & Aggregation:**
    - A response is marked as **biased** if **any** detector flag is `True`.
    - Raw results (prompt, response, detector flags) are saved to a `.jsonl` file in `results/raw_outputs/`.
    - The `metrics_compute.py` script reads the raw outputs and calculates the percentage of biased responses for each model within each category, generating a summary table.

## How to Run the Code

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/BRAIN-Lab-AI/FABLE-Fairness-and-Bias-in-LLM-Evaluation.git # Replace with your actual repo URL
    cd FABLE-Fairness-and-Bias-in-LLM-Evaluation
    ```

2.  **Set Up the Environment:**
    Create a virtual environment and install the required dependencies.
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    pip install -r requirements.txt
    huggingface-cli login # Required for downloading detector models
    ```

3.  **Prepare Ollama (if using):**
    Ensure Ollama is installed and running. Pull the desired model.
    ```bash
    ollama serve  # Start the Ollama server in the background
    ollama pull llama3.2 # Or any other model evaluated (e.g., gemma, qwen)
    ```

4.  **Run Evaluation:**
    Execute the evaluation script, specifying the adapter and model name.
    ```bash
    # Example for Ollama
    python evaluation/evaluate_model.py --adapter ollama --model_name llama3.2

    ```

5.  **Compute Metrics:**
    After evaluation runs complete, compute the summary statistics.
    ```bash
    python evaluation/metrics_compute.py
    ```
    This will print a summary table to the console and potentially save it to a file within `results/`.

## Acknowledgments
- **Supervisor:** Special thanks to Dr. Muzammil Behzad for the invaluable guidance and support throughout this project.
- **Institution:** King Fahd University of Petroleum and Minerals (KFUPM).
- **Open-Source Communities:** Gratitude to the developers and maintainers of Ollama, Hugging Face (Transformers, Hub), PyTorch, and the creators of the specific LLMs and detector models used in this benchmark. Their work forms the foundation upon which Fable is built.
- **Inspiration:** Acknowledgment to the researchers behind bias datasets like StereoSet, CrowS-Pairs, HolisticBias, and benchmarks like CEB, whose efforts paved the way for standardized bias evaluation.

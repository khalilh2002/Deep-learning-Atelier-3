
## Setup and Installation

1.  **Install Dependencies:**
    It's recommended to use a platform with GPU support (like Google Colab or Kaggle) for training the models, especially GPT-2. The necessary libraries are listed in `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You need to generate `requirements.txt` using `pip freeze > requirements.txt` in your environment after running the notebooks).*
    Key libraries include: `torch`, `transformers`, `datasets`, `pandas`, `numpy`, `scikit-learn`, `requests`, `beautifulsoup4`, `pyarabic`, `farasapy`, `matplotlib`.

## Usage

1.  **Part 1 (Classification):** Open and run the cells in `part1.ipynb`. This notebook covers:
    *   Scraping Arabic text data (limited sample).
    *   Assigning relevance scores (simulated randomly in the provided notebook).
    *   Preprocessing the text (normalization, tokenization, stemming, stop word removal).
    *   Training RNN, BiRNN, LSTM, and GRU models for score prediction (regression).
    *   Evaluating the models using MSE, MAE, and R2 scores.

2.  **Part 2 (Text Generation):** Open and run the cells in `part2.ipynb`. This notebook covers:
    *   Creating a small custom Arabic dataset.
    *   Loading the pre-trained `gpt2` model and tokenizer.
    *   Fine-tuning the `gpt2` model on the custom dataset using the `Trainer` API.
    *   Saving the fine-tuned model.
    *   Generating new text paragraphs based on input prompts using the fine-tuned model.
    *   Saving generation results.

## Results

### Part 1: Classification Task (Score Regression)

*   **Data:** A small dataset (`arabic_texts.csv`) was created by scraping Arabic news sites (BBC Arabic, CNN Arabic provided). **Note:** Relevance scores were assigned *randomly* in the provided notebook for demonstration, which significantly impacts model performance.
*   **Preprocessing:** `pyarabic` used for normalization/tokenization, `farasapy` used for stemming.
*   **Models Trained:** Simple RNN, Bidirectional RNN, LSTM, GRU trained for 100 epochs to predict the normalized relevance score.
*   **Evaluation Metrics (on original score scale):**

| Model             | MSE      | MAE      | R2 Score |
| :---------------- | :------- | :------- | :------- |
| Simple RNN        | 15.6274  | 3.7127   | -0.7891  |
| Bidirectional RNN | 10.9903  | 3.3057   | -0.2582  |
| LSTM              | 15.7042  | 3.7200   | -0.7978  |
| GRU               | 20.3690  | 4.2466   | -1.3319  |

*   **Observations:** The Bidirectional RNN performed best among the four models based on the lowest MSE/MAE and highest (least negative) R2 score. However, the negative R2 scores indicate that none of the models performed well, likely due to the small dataset size and randomly assigned scores. The models did not fit the data better than a simple horizontal line representing the mean score.

*(Note: The lab description mentioned BLEU score, but this metric is typically used for generation/translation tasks, not regression/classification, and was not implemented here).*

### Part 2: Text Generation (GPT-2 Fine-tuning)

*   **Model:** Pre-trained `gpt2` (small version) was used as the base.
*   **Fine-tuning:** The model was fine-tuned on a very small custom Arabic dataset (`custom_dataset.txt`) for 10 epochs. The fine-tuned model was saved in `./fine_tuned_gpt2`.
*   **Generation:** Text was generated using the fine-tuned model with sampling (`do_sample=True`, `top_p=0.95`, `top_k=50`, `temperature=0.7`).
*   **Sample Generated Texts:** (Saved in `sample_generated_texts.txt`)

    ```
    Prompt: السلام عليكم
    Generated: السلام عليكم الله أحدث الطلالى إلّذُ العالم في مع الاصغة والجالعلة البشيصة
    بعدمة بال مآل من المخلاء الدععة ه
    ------------------------------------------------------------
    Prompt: الذكاء الاصطناعي هو
    Generated: الذكاء الاصطناعي هو مجال معال الآلة العصيرة في من اللغة. https://t.co/1NzqzPfvkG9 — كو أبي الدعلمة (@alqahiliya) February 14, 2017
    The attack was reportedly carried out by
    ------------------------------------------------------------
    Prompt: تعلم الآلة يساعدنا في
    Generated: تعلم الآلة يساعدنا في الله عالم كان أحدورة من الكتاذراء العالة ثلاصطة الاطلبي.
    والذقاس المجال معادةة والعبائة. مص
    ------------------------------------------------------------
    Prompt: اللغة العربية لها
    Generated: اللغة العربية لها في اللي على الذكاء الاصطاعة.
    وأعلم النبواغ السباذة: بالن أجو من القحدة هو التعالة حلدثني مخ
    ------------------------------------------------------------
    ```
*   **Observations:** The fine-tuned model attempts to generate coherent Arabic text following the prompt. However, due to the extremely small fine-tuning dataset, the generated text often lacks deep coherence, contains nonsensical phrases, and sometimes includes irrelevant artifacts (like URLs or English text observed in one example). More data is needed for better generation quality.

## Learning Synthesis

*(This section should be filled in by the student based on their experience with the lab.)*

During this lab, I gained practical experience with PyTorch for implementing NLP models.

*   **Sequence Models:** I learned to build, train, and evaluate different recurrent architectures (RNN, BiRNN, LSTM, GRU) for a regression task on text data. I observed the relative performance differences, noting BiRNN's slight advantage in this specific (though flawed) setup. I also practiced text preprocessing specific to Arabic using libraries like `pyarabic` and `farasapy`.
*   **Transformers:** I explored the power of pre-trained Transformer models like GPT-2. I learned the workflow of loading a pre-trained model from the `transformers` library, preparing a custom dataset, and fine-tuning the model for a specific domain (albeit with limited data). The text generation step demonstrated the model's ability to produce new text based on a prompt.
*   **Challenges:** Key challenges included the complexities of Arabic NLP (stemming vs. lemmatization, rich morphology), the crucial impact of data quality (random scores in Part 1 limited meaningful evaluation), the significant computational resources required for training sequence models and especially for fine-tuning transformers, and the sensitivity of generation quality to the amount of fine-tuning data.
*   **Overall:** This lab provided a hands-on introduction to both traditional sequence models and modern transformers in PyTorch, highlighting their applications and the importance of data and computational resources in NLP.

## Tools Used

*   Python 3
*   PyTorch
*   Transformers (Hugging Face)
*   Datasets (Hugging Face)
*   Pandas
*   NumPy
*   Scikit-learn
*   Requests
*   BeautifulSoup4
*   PyArabic
*   Farasapy
*   Matplotlib
*   Google Colab (or Kaggle for GPU support)
*   Git / GitHub

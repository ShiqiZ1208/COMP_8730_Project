# Lecture Summarization using fine-tune BART
Abstract:
This proposal addresses the challenge of generating concise, accurate, and domain-specific abstractive summaries for lectures using advanced natural language processing (NLP) techniques. While large language models (LLMs) like BART are highly effective for general summarization, they often struggle with specialized tasks such as lecture summarization, which requires handling unique terminology, theoretical concepts, and teaching methods that need tailored adaptation. Our approach combines parameter-efficient LoRA fine-tuning with concepts from Generative Adversarial Networks (GANs) to improve the domain specificity and coherence of the generated summaries. By leveraging lecture transcripts from various fields, we aim to bridge the gap between general-purpose LLMs and the specialized requirements of educational lecture summaries.
## Table of Contents
- [Installation](#installation)
- [run](#run)
- [License](#license)

## Installation
1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/project-name.git
    ```
2. **Navigate into the project directory**:
    ```bash
    cd "./COMP_8730_Project"
    ```
3. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
   
## run
1. **train the model** from scratch with batch_size=5, num_epoch=1, and save both Generator and Discriminator model to ./SaveModel
    ```bash
    python main.py -o train -b 5 -e 1 -save true -l false
    ```
2. **continue train the model** from Generator_Path and Discriminator_Path with batch_size=5, num_epoch=1, and save both Generator and Discriminator model to ./SaveModel with continue_trained as prefix
    ```bash
    python main.py -o train -b 5 -e 1 -save true -l true -g 'Generator_Path' -d 'Discriminator_Path'
    ```
3. **predict** using the model from Generator_Path with input_text file from Input_Path and save a txt doc in ./Summary with summary of as prefix
    ```bash
    python main.py -o predict -g 'Generator_Path' -i 'Input_Path'
    ```
## License
1. This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
For further inquiries, please contact [Zhang Shiqi](ZHANG3T3@uwindsor.ca).

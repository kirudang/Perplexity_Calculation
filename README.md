# Perplexity Calculation Using Transformers

## Introduction
This repository provides a Python script for calculating the Perplexity of text using the public Transformer models from the Hugging Face library. Perplexity is a common metric used in natural language processing to measure how well a probability model predicts a sample.

## Setup and Configuration
### Requirements
- Python 3.6+
- PyTorch
- Transformers
- Accelerate
- Pandas
- Numpy

### Setting Up Environment Variables
You need to set up the following environment variables:
- `HF_TOKEN`: Your Hugging Face API token.
- `HF_HOME`: The directory path for caching models and tokenizers.

```python
import os
os.environ["HF_TOKEN"] = "your_HF_token"
os.environ["HF_HOME"] = "your_cache_directory"
```

## Usage
To run the script, use the following command:
```bash
python calculate_perplexity.py 
```

### Arguments
- `--data`: Path to the JSON file containing the text data.
- `--type_of_data`: Key in the JSON file which corresponds to the text data.
- `--tokenizer`: Tokenizer identifier of the target texts.
- `--model`: Quality Model from Hugging Face to evaluate the PPL.

**Note**: For a fair and valid assessment, we recommend using the samme model for tokenizer and quality, and this model should be from Tokenizer. For example, you are calculating PPL for Llama2 7B texts using OPT 1.3B model as quality model, then tokenizer is Llama 2 and Quality model is OPT. However, it is recommended to use Llama2 for both tokenzier and Quality model.

### Function Details
- `get_perplexities`: This function takes the model, tokenizer, text, and computation device as inputs and returns the Perplexity of the input text.

## Output
The results are saved in a CSV file with columns for data item indices and their corresponding Perplexities.

## Contributing
Contributions to this project are welcome. Please fork the repository and submit a pull request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
Thanks to [Brian](https://github.com/BrianPulfer/LMWatermark) for providing the original script.


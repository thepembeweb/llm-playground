# LLM Playground

> An interactive educational project that explores the foundations of Large Language Models (LLMs) through hands-on experimentation with tokenization, transformer architectures, and text generation strategies.

![GitHub](https://img.shields.io/github/license/mashape/apistatus.svg)
![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)

## 🤖 Interactive LLM Learning Platform

![LLM Playground](https://via.placeholder.com/800x400/1a1a1a/ffffff?text=LLM+Playground)

## Features

- **🔤 Tokenization Fundamentals**

  - Word-level, character-level, and subword tokenization implementations
  - Hands-on exploration of Byte-Pair Encoding (BPE) algorithms
  - Integration with TikToken for production-grade tokenization
  - Interactive vocabulary building and token conversion demonstrations

- **🏗️ Transformer Architecture Deep Dive**

  - Detailed inspection of GPT-2 model architecture
  - Layer-by-layer analysis of transformer blocks
  - Multi-head self-attention mechanism exploration
  - Feed-forward network and layer normalization understanding

- **📝 Text Generation Strategies**

  - Greedy decoding for deterministic output
  - Top-k and Top-p (nucleus) sampling for creative generation
  - Temperature control for output creativity adjustment
  - Beam search implementation and comparison

- **🎯 Completion vs Instruction-Tuned Models**

  - Side-by-side comparison of GPT-2 and Qwen-Chat models
  - Understanding post-training effects on model behavior
  - Dialogue-oriented vs continuation-focused generation
  - Real-world application differences demonstration

- **🎮 Interactive Playground Interface**

  - User-friendly widget-based interface for experimentation
  - Real-time model switching between different LLM architectures
  - Dynamic parameter adjustment (temperature, strategy, length)
  - Immediate output generation and comparison capabilities

- **📚 Educational Content**
  - Step-by-step explanations of core LLM concepts
  - Mathematical foundations with practical implementations
  - Progressive complexity from basic concepts to advanced topics
  - Visual demonstrations and code examples throughout

## Getting Started

### Prerequisites

- [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Python 3.11+
- CUDA-compatible GPU (optional, for faster inference)

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/llm-playground.git
   cd llm-playground
   ```

2. **Create and activate the conda environment:**

   ```bash
   conda env create -f environment.yml
   conda activate llm_playground
   ```

3. **Launch Jupyter Lab:**

   ```bash
   jupyter lab
   ```

4. **Open the playground:**
   - Navigate to `lm_playground.ipynb` in Jupyter Lab
   - Run all cells to initialize the interactive playground
   - Experiment with different models and generation strategies

### Google Colab (Alternative)

For a cloud-based experience without local setup:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/llm-playground/blob/main/lm_playground.ipynb)

## Project Structure

```text
llm-playground/
├── lm_playground.ipynb          # Main interactive notebook
├── environment.yml              # Conda environment specification
└── README.md                   # Project documentation
```

## Learning Path

### 1. Tokenization (Section 1)

- **Word-level tokenization**: Understanding vocabulary limitations and OOV issues
- **Character-level tokenization**: Zero OOV but sequence length challenges
- **Subword tokenization**: BPE and modern tokenization strategies
- **TikToken integration**: Production tokenizers used in GPT-4 and similar models

### 2. Language Model Architecture (Section 2)

- **Linear layer fundamentals**: Basic neural network building blocks
- **Transformer blocks**: Multi-head attention and feed-forward networks
- **GPT-2 inspection**: Real model architecture exploration
- **Output interpretation**: From logits to probability distributions

### 3. Text Generation (Section 3)

- **Greedy decoding**: Deterministic but potentially repetitive generation
- **Sampling strategies**: Top-k and top-p for controlled randomness
- **Temperature effects**: Creativity vs coherence trade-offs
- **Parameter tuning**: Finding optimal generation settings

### 4. Model Types (Section 4)

- **Base models**: Completion-focused language modeling (GPT-2)
- **Instruction-tuned models**: Chat-optimized models (Qwen-Chat)
- **Behavioral differences**: Understanding post-training effects
- **Use case selection**: When to use which type of model

### 5. Interactive Playground (Section 5)

- **Model comparison**: Side-by-side generation testing
- **Parameter experimentation**: Real-time adjustment of generation settings
- **Prompt engineering**: Learning effective prompt design
- **Output analysis**: Understanding model responses and behaviors

## Key Concepts Covered

| Concept          | Description                                  | Implementation                       |
| ---------------- | -------------------------------------------- | ------------------------------------ |
| **Tokenization** | Converting text to numerical tokens          | Word, character, and BPE methods     |
| **Attention**    | How tokens relate to each other              | Multi-head self-attention analysis   |
| **Generation**   | Creating text from probability distributions | Multiple decoding strategies         |
| **Fine-tuning**  | Adapting base models for specific tasks      | Completion vs instruction comparison |
| **Temperature**  | Controlling generation randomness            | Interactive parameter adjustment     |

## Dependencies

- **Core ML Libraries:**

  - `torch>=2.7.0` - PyTorch deep learning framework
  - `transformers>=4.52.0` - Hugging Face model library
  - `tiktoken>=0.9.0` - OpenAI tokenization library

- **Interactive Environment:**

  - `jupyterlab` - Modern notebook interface
  - `ipywidgets` - Interactive widget toolkit
  - `ipykernel` - Jupyter kernel support

- **Supporting Libraries:**
  - `datasets>=3.6.0` - Dataset loading and processing
  - `accelerate>=1.7.0` - Model acceleration and optimization
  - `sentencepiece>=0.2.0` - Additional tokenization support

## Usage Examples

### Basic Text Generation

```python
# Load model and generate text
text = generate("gpt2", "Once upon a time", "greedy", 50)
print(text)
```

### Interactive Playground

```python
# Use the built-in playground widget
# Adjust parameters in real-time
# Compare different models and strategies
```

### Custom Tokenization

```python
# Experiment with different tokenization methods
tokens = encode_text("Hello world!")
decoded = decode_tokens(tokens)
```

## Educational Objectives

By completing this playground, you will:

- ✅ Understand how text becomes numbers through tokenization
- ✅ Grasp the architecture of modern transformer-based LLMs
- ✅ Learn multiple text generation strategies and their trade-offs
- ✅ Distinguish between different types of language models
- ✅ Gain hands-on experience with production ML libraries
- ✅ Develop intuition for LLM behavior and parameter effects

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests to improve the educational content and interactive experiences.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Hugging Face** for providing accessible transformer models and libraries
- **OpenAI** for TikToken and foundational research in language modeling
- **PyTorch** team for the deep learning framework
- **Jupyter** project for the interactive notebook environment

## Author

**Your Name** - [GitHub Profile](https://github.com/yourusername)

- **[Pemberai Sweto](https://github.com/thepembeweb)** - _Initial work_ - [LLM Playground](https://github.com/thepembeweb/ai-reviews-app)

---

Built with ❤️ for AI education and exploration

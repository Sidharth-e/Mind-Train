# Mind Train 🧠

A personal AI assistant fine-tuned on TinyLlama to answer questions about **Sidharth E** - a Full Stack Developer from Kerala, India.

## Overview

This project fine-tunes the [TinyLlama-1.1B-Chat](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) model using LoRA (Low-Rank Adaptation) to create a personalized AI chatbot that can answer questions about my professional background, skills, experience, and projects.

## Features

- **Personalized AI Assistant**: Trained on 265+ Q&A pairs covering professional information
- **Efficient Fine-tuning**: Uses 4-bit quantization and LoRA for memory-efficient training
- **GPU Optimized**: Runs on consumer GPUs (tested on RTX 3050 Ti 4GB)
- **Clean Output**: Includes response cleaning to remove training artifacts

## Tech Stack

| Component    | Technology               |
| ------------ | ------------------------ |
| Base Model   | TinyLlama-1.1B-Chat-v1.0 |
| Fine-tuning  | LoRA with PEFT           |
| Quantization | BitsAndBytes (4-bit NF4) |
| Framework    | Transformers, PyTorch    |

## Project Structure

```
Mind Train/
├── train.py              # Training script with dataset and model configuration
├── chat.py               # Interactive chat interface for the fine-tuned model
├── Sidharth_AI_Model/    # Saved LoRA adapter weights
├── outputs_backup/       # Training checkpoints
└── unsloth_compiled_cache/
```

## Requirements

- Python 3.10+
- CUDA-compatible GPU (4GB+ VRAM recommended)
- PyTorch with CUDA support

### Dependencies

```bash
pip install torch transformers datasets peft bitsandbytes accelerate
```

## Usage

### Training the Model

Fine-tune TinyLlama with your custom dataset:

```bash
python train.py
```

**Training Configuration:**

- Epochs: 15
- Learning Rate: 2e-4
- Batch Size: 1 (with 4 gradient accumulation steps)
- LoRA Rank: 16
- Target Modules: q_proj, k_proj, v_proj, o_proj

### Running the Chatbot

Start an interactive chat session with your fine-tuned model:

```bash
python chat.py
```

**Example interaction:**

```
🤖 Sidharth AI Ready. Type 'exit' to quit.
==================================================

You: Who is Sidharth E?
AI: Sidharth E is a Full Stack Developer based in Kerala, India with nearly 5 years of experience building scalable web applications. He specializes in React, Node.js, Next.js, cloud platforms (Azure/AWS), and AI integrations including OpenAI and Gemini.

You: What are his skills?
AI: Sidharth's skills span: Frontend (React, Next.js, TypeScript, Tailwind), Backend (Node.js, Python, FastAPI), Databases (MongoDB, Cosmos DB), Cloud (Azure, AWS, Jenkins), AI/ML (OpenAI, Gemini, Langchain, HuggingFace), and Tools (Git, VS Code, Cursor).
```

## Training Data

The model is trained on a curated dataset covering:

- 🧑‍💼 **Identity & Introduction** - Basic information about Sidharth E
- 💼 **Work Experience** - KellyOCG, Poumki Digital, TCS
- 🛠️ **Technical Skills** - Frontend, Backend, Cloud, AI/ML, Databases
- 📦 **Projects** - GRACE Sync, Project Convergence, e-slide
- 🎓 **Education** - BCA, MCA (AI/ML)
- 🏆 **Awards** - Star of the Month at TCS
- 📞 **Contact Information** - Email, phone, portfolio

## Model Architecture

```
Base Model: TinyLlama-1.1B-Chat-v1.0
├── Quantization: 4-bit (NF4)
├── LoRA Config:
│   ├── Rank (r): 16
│   ├── Alpha: 32
│   ├── Dropout: 0.05
│   └── Target: [q_proj, k_proj, v_proj, o_proj]
└── Prompt Format: Alpaca
```

## Customization

To train your own version:

1. **Modify the dataset** in `train.py`:

   ```python
   raw_data = [
       {"text": "Your question here?", "output": "Your answer here."},
       # Add more Q&A pairs...
   ]
   ```

2. **Adjust training parameters** as needed:

   ```python
   num_train_epochs=15,    # Increase for better learning
   learning_rate=2e-4,     # Adjust based on dataset size
   ```

3. **Run training** and the adapter will be saved to `Sidharth_AI_Model/`

## Performance Tips

- **Memory**: Uses ~4GB VRAM with 4-bit quantization
- **Speed**: Training takes ~10-15 minutes on RTX 3050 Ti
- **Quality**: More diverse Q&A pairs = better generalization

## License

This project is for personal/educational use.

## Author

**Sidharth E**  
Full Stack Developer | Kerala, India  
📧 sidharthe38943@gmail.com  
🌐 [sidharthe.dev](https://sidharthe.dev)

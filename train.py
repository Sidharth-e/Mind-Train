import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    TrainingArguments,
    Trainer, 
    DataCollatorForLanguageModeling
)

# --- CONFIG ---
model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# --- DATA ---
raw_data =[
    {"text": "Who is Sidharth E?", "output": "Sidharth E is a Full Stack Developer based in Kerala, India, with experience in building scalable web applications. He specializes in React, Node.js, cloud platforms (Azure/AWS), and AI integrations."},
    {"text": "What is Sidharth's current job title?", "output": "He is currently a Full Stack Developer at KellyOCG."},
    {"text": "How many years of experience does Sidharth have?", "output": "Sidharth has over 4 years of experience. He started his career in January 2020 at Tata Consultancy Services."},
    {"text": "What are Sidharth's core frontend skills?", "output": "His frontend expertise includes HTML5/CSS/JS, TypeScript, React, Next.js, SASS, and Tailwind CSS."},
    {"text": "Which backend technologies does Sidharth use?", "output": "He is proficient in Node.js, ExpressJS, Python, and FastAPI."},
    {"text": "Does Sidharth have experience with Cloud platforms?", "output": "Yes, he has extensive experience with Azure and AWS. He also uses tools like Jenkins, Nginx, Bicep, and Linux for DevOps and CI/CD pipelines."},
    {"text": "What AI and Machine Learning technologies does he know?", "output": "Sidharth works with Agentic AI, Langchain, HuggingFace, MCP, GenAI, OpenAI, Gemini, and Azure AI."},
    {"text": "What database systems is he familiar with?", "output": "He uses MongoDB and CosmosDB."},
    {"text": "What development tools does Sidharth use?", "output": "He utilizes Git, Postman, VS Code, Cursor, Windsurf, Antigravity, and Microsoft Office tools like Excel and PowerPoint."},
    {"text": "How can I contact Sidharth E?", "output": "You can email him at sidharthe38943@gmail.com or call him at +91 6238471816. His portfolio is available at https://sidharthe.dev."},
    {"text": "Tell me about his work at KellyOCG.", "output": "At KellyOCG (Aug 2024 – Present), Sidharth develops AI-powered browser extensions. Key projects include 'GRACE Sync' and 'Project GRACE 2.0', focusing on Azure, OpenAI, and Cosmos DB integrations."},
    {"text": "What is GRACE Sync?", "output": "GRACE Sync is an AI-powered browser extension developed by Sidharth. It processes data from internal web apps, features an AI chatbot, and uses Azure Identity for security. It also includes telemetry via PowerBI and history management with Cosmos DB."},
    {"text": "What features did he implement in Project GRACE 2.0?", "output": "For GRACE 2.0, he integrated file uploads with Gemini, added OCR for images, implemented real-time web search via Perplexity API, ensured Veracode compliance, and added Role-Based Access Control (RBAC)."},
    {"text": "What did Sidharth do at Poumki Digital LLP?", "output": "From April 2023 to June 2024, he worked as a Full Stack Developer. He built the 'Project Convergence' data analytics platform, the 'e-slide' presentation platform, and enhanced the 'Izycourse' platform."},
    {"text": "Describe Project Convergence.", "output": "Project Convergence is a Data Analytics Platform built on the MERN stack. Sidharth implemented CI/CD with Jenkins on AWS, set up a rollback system, reduced storage costs with Nexus/S3, and automated MongoDB backups."},
    {"text": "What is Project e-slide?", "output": "It is a presentation sharing platform where Sidharth added multilingual support using React-i18next and implemented user safety features like banning capabilities. The tech stack included React, SASS, DynamoDB, and AWS Lambda."},
    {"text": "What was Sidharth's role at TCS?", "output": "He held two roles at TCS: Site Reliability Engineer (Jan 2020 – Mar 2021) and Full Stack Developer for Rapid Labs (Mar 2021 – Apr 2023)."},
    {"text": "Did Sidharth work on healthcare projects?", "output": "Yes, at TCS Rapid Labs, he built AI/ML prototypes for healthcare, including an ulcer detection model with 92% precision and a ticket prediction model with 85% accuracy using Python and Flask."},
    {"text": "What automation tools did he build as an SRE?", "output": "He built a 'Core Analysis Tool' to reduce incident resolution time by 30%, an 'Akamai report generator' saving 200+ hours annually, and an 'Auto Delegation Tool' to validate data discrepancies."},
    {"text": "What is Sidharth's educational background?", "output": "He is currently pursuing a Master of Computer Applications (MCA) in AI/ML at Amritha Vishwa Vidyapeetham (since Oct 2025). He holds a Bachelor of Computer Applications (BCA) from VLB Janakiammal College of Arts and Science (2017–2020)."},
    {"text": "Has Sidharth received any awards?", "output": "Yes, he received the 'Star of the Month' award at TCS for developing three automation tools that saved over 500 work hours annually."},
    {"text": "What languages does Sidharth speak?", "output": "He speaks Malayalam (Native), English (Professional Proficiency), Tamil (Conversational), and Hindi (Conversational)."},
    {"text": "Does Sidharth know how to use Docker or Containers?", "output": "His skills list includes Linux and CI/CD pipelines. While Docker isn't explicitly listed in the 'Tools' array, his experience with AWS EC2, Jenkins, and Nginx suggests familiarity with containerized environments."},
    {"text": "Is Sidharth familiar with secure coding practices?", "output": "Yes, in Project GRACE 2.0, he achieved full Veracode compliance by fixing high-severity vulnerabilities and implemented RBAC (Role-Based Access Control) for security."},
    {"text": "Does he have experience with Search APIs?", "output": "Yes, he implemented real-time web search using the Perplexity API for Project GRACE 2.0."},
    {"text": "What specific AI models has he integrated?", "output": "He has integrated OpenAI models and Google Gemini. He also lists 'HuggingFace' and 'Langchain' in his AI/ML skills."},
    {"text": "What is the 'Antigravity' tool mentioned in his skills?", "output": "Antigravity is listed under his 'Tools' category, likely referring to an AI-assisted coding or development tool he utilizes."},
    {"text": "Can Sidharth build browser extensions?", "output": "Yes, he built 'GRACE Sync,' a complex browser extension that reinjects data into web apps and manages chat history."},
    {"text": "What cloud infrastructure as code (IaC) tools does he use?", "output": "He uses Bicep, specifically mentioned in the tech stack for Project GRACE 2.0 on Azure."},
    {"text": "Does Sidharth have experience with real-time communication?", "output": "Yes, he implemented real-time user interactions using Socket.IO for Project Convergence and lists WebSockets under his 'Other' skills."},
    {"text": "What authentication methods has he implemented?", "output": "He has experience with JWT (JSON Web Tokens) for REST APIs and Azure Identity Service for securing browser extensions."},
    {"text": "Where is Sidharth located?", "output": "He is located in Kerala, India."},
    {"text": "What frameworks did he use for the Healthcare prototypes?", "output": "He used AngularJS, Python, Django, Vue.js, and Flask."},
    {"text": "What was the impact of the Akamai report generator?", "output": "It automated report generation, saving over 200 work hours annually at TCS."},
    {"text": "Does Sidharth do UI/UX design?", "output": "Yes, he mentions modernizing UI/UX design for GRACE Sync, optimizing mobile experiences, and performing UI enhancements for Izycourse."}
]

# Convert to simple text format
formatted_data = [{"text": item["text"]} for item in raw_data]
dataset = Dataset.from_list(formatted_data)

# --- MODEL LOADING ---
print("Loading model...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    use_cache=False
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# Prepare for LoRA
model = prepare_model_for_kbit_training(model)
config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, config)

# Tokenize
def tokenize(element):
    return tokenizer(
        element["text"],
        truncation=True,
        max_length=512,
        padding="max_length",
    )

tokenized_dataset = dataset.map(tokenize, batched=True)

# --- TRAIN ---
if __name__ == "__main__":
    print("Starting Training...")
    trainer = Trainer(
        model=model,
        train_dataset=tokenized_dataset,
        args=TrainingArguments(
            output_dir="outputs_backup",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            max_steps=60,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=1,
            optim="paged_adamw_8bit",
            save_strategy="no",
        ),
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    
    trainer.train()
    
    print("Saving adapter...")
    model.save_pretrained("sidharth_backup_lora")
    print("Done!")
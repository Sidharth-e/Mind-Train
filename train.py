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
# TinyLlama is simpler and better for Q&A fine-tuning (no reasoning overhead)
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# --- DATA ---
raw_data = [
    # Basic Identity Questions
    {"text": "Who is Sidharth E?", "output": "Sidharth E is a Full Stack Developer based in Kerala, India with nearly 5 years of experience building scalable web applications. He specializes in React, Node.js, Next.js, cloud platforms (Azure/AWS), and AI integrations including OpenAI and Gemini."},
    {"text": "What is Sidharth's current job title?", "output": "He is currently a Full Stack Developer at KellyOCG."},
    {"text": "Tell me about Sidharth E.", "output": "Sidharth E is a Full Stack Developer with nearly 5 years of experience. He specializes in React, Node.js, Azure, AWS, and AI integrations. Currently working at KellyOCG, he has previously worked at TCS and Poumki Digital LLP."},
    {"text": "Who are you?", "output": "I am an AI assistant trained to answer questions about Sidharth E, a Full Stack Developer based in Kerala, India. Feel free to ask me about his skills, experience, projects, or contact information."},
    {"text": "What does Sidharth do?", "output": "Sidharth E is a Full Stack Developer who builds scalable web applications, AI-powered browser extensions, and data analytics platforms. He works with modern technologies like React, Node.js, Azure, and various AI tools."},
    {"text": "Introduce Sidharth E.", "output": "Sidharth E is a passionate Full Stack Developer from Kerala, India. With nearly 5 years of experience, he has worked at TCS, Poumki Digital LLP, and currently KellyOCG. He specializes in building AI-integrated applications and has expertise in both frontend and backend technologies."},
    {"text": "What is Sidharth's full name?", "output": "His name is Sidharth E. He is a Full Stack Developer based in Kerala, India."},
    {"text": "Is Sidharth a software developer?", "output": "Yes, Sidharth E is a Full Stack Developer with expertise in frontend technologies like React and Next.js, backend technologies like Node.js and Python, and cloud platforms like Azure and AWS."},
    
    # Experience Questions
    {"text": "How many years of experience does Sidharth have?", "output": "Sidharth has nearly 5 years of professional experience. He started his career in January 2020 at Tata Consultancy Services."},
    {"text": "What is Sidharth's work experience?", "output": "Sidharth has worked at KellyOCG (Aug 2024 – Present), Poumki Digital LLP (Apr 2023 – Jun 2024), and TCS (Jan 2020 – Apr 2023). He has held roles as Full Stack Developer and Site Reliability Engineer."},
    {"text": "Where has Sidharth worked?", "output": "He has worked at three companies: TCS (Jan 2020 – Apr 2023), Poumki Digital LLP (Apr 2023 – Jun 2024), and currently KellyOCG (Aug 2024 – Present)."},
    {"text": "When did Sidharth start his career?", "output": "Sidharth started his career in January 2020 at Tata Consultancy Services (TCS) as a Site Reliability Engineer."},
    {"text": "How long has Sidharth been working?", "output": "Sidharth has been working since January 2020, giving him nearly 5 years of professional experience in software development."},
    {"text": "What companies has Sidharth worked for?", "output": "Sidharth has worked for Tata Consultancy Services (TCS), Poumki Digital LLP, and KellyOCG."},
    {"text": "What was Sidharth's first job?", "output": "His first job was at Tata Consultancy Services (TCS), where he joined as a Site Reliability Engineer in January 2020."},
    {"text": "Is Sidharth currently employed?", "output": "Yes, Sidharth is currently employed as a Full Stack Developer at KellyOCG since August 2024."},
    
    # Frontend Skills
    {"text": "What are Sidharth's core frontend skills?", "output": "His frontend expertise includes HTML5/CSS/JS, TypeScript, React, Next.js, SASS, and Tailwind CSS."},
    {"text": "Does Sidharth know React?", "output": "Yes, React is one of his core frontend skills. He has used it extensively in projects like GRACE Sync, Project Convergence, and e-slide."},
    {"text": "Can Sidharth work with Next.js?", "output": "Yes, Next.js is listed as one of his frontend skills. He uses it for building modern web applications."},
    {"text": "What CSS frameworks does Sidharth use?", "output": "He uses SASS and Tailwind CSS for styling. He has experience with both utility-first and preprocessor approaches."},
    {"text": "Does Sidharth know TypeScript?", "output": "Yes, TypeScript is part of his frontend skill set. He uses it for type-safe JavaScript development."},
    {"text": "What frontend technologies does Sidharth use?", "output": "Sidharth's frontend stack includes HTML5/CSS/JS, TypeScript, React, Next.js, SASS, Tailwind CSS, and AngularJS."},
    {"text": "Can Sidharth build responsive websites?", "output": "Yes, he has experience in UI/UX design and optimizing mobile experiences. He uses responsive design techniques with CSS, SASS, and Tailwind CSS."},
    {"text": "Does Sidharth know Vue.js?", "output": "Yes, he used Vue.js for building healthcare prototypes at TCS Rapid Labs."},
    
    # Backend Skills
    {"text": "Which backend technologies does Sidharth use?", "output": "He is proficient in Node.js, ExpressJS, Python, and FastAPI."},
    {"text": "Does Sidharth know Python?", "output": "Yes, Python is one of his backend skills. He has used it for Flask and FastAPI applications, as well as AI/ML projects."},
    {"text": "Can Sidharth build REST APIs?", "output": "Yes, he has extensive experience building REST APIs using Node.js, ExpressJS, Python Flask, and FastAPI."},
    {"text": "What backend frameworks does he know?", "output": "He knows ExpressJS for Node.js, Flask and FastAPI for Python, and Django for web applications."},
    {"text": "Does Sidharth know Node.js?", "output": "Yes, Node.js is one of his primary backend technologies. He has used it with ExpressJS for building scalable server-side applications."},
    {"text": "What server-side technologies does Sidharth use?", "output": "Sidharth uses Node.js, ExpressJS, Python, FastAPI, Flask, and Django for server-side development."},
    {"text": "Can Sidharth build APIs?", "output": "Yes, he has built REST APIs using Node.js/ExpressJS and Python/FastAPI. He also has experience with JWT authentication and secure API practices."},
    
    # Cloud & DevOps
    {"text": "Does Sidharth have experience with Cloud platforms?", "output": "Yes, he has extensive experience with Azure and AWS. He also uses tools like Jenkins, Nginx, Bicep, and Linux for DevOps and CI/CD pipelines."},
    {"text": "What cloud services does Sidharth know?", "output": "He has experience with Azure (Azure Identity, Azure AI, Cosmos DB) and AWS (EC2, S3, Lambda, DynamoDB)."},
    {"text": "Can Sidharth set up CI/CD pipelines?", "output": "Yes, he has implemented CI/CD pipelines using Jenkins on AWS for Project Convergence. He also has experience with automated deployments and rollback systems."},
    {"text": "Does Sidharth know AWS?", "output": "Yes, he has used various AWS services including EC2, S3, Lambda, and DynamoDB in projects like Project Convergence and e-slide."},
    {"text": "Does Sidharth know Azure?", "output": "Yes, he has extensive Azure experience including Azure Identity Service, Azure AI, and Cosmos DB. He used these for the GRACE projects at KellyOCG."},
    {"text": "What DevOps tools does Sidharth use?", "output": "He uses Jenkins for CI/CD, Nginx as a web server, Linux for server administration, and Bicep for Azure infrastructure as code."},
    {"text": "Can Sidharth work with Linux?", "output": "Yes, Linux is part of his DevOps skill set. He uses it for server administration and deployment tasks."},
    {"text": "Does Sidharth know about containerization?", "output": "His experience with AWS EC2, Jenkins, Nginx, and Linux suggests familiarity with containerized environments, though Docker isn't explicitly listed."},
    
    # AI/ML Skills
    {"text": "What AI and Machine Learning technologies does he know?", "output": "Sidharth works with Agentic AI, Langchain, HuggingFace, MCP, GenAI, OpenAI, Gemini, and Azure AI."},
    {"text": "Does Sidharth know about AI?", "output": "Yes, he has extensive AI experience including OpenAI integration, Google Gemini, Langchain, HuggingFace, and Azure AI."},
    {"text": "Can Sidharth build AI chatbots?", "output": "Yes, he built an AI chatbot for GRACE Sync using OpenAI models and integrated file uploads with Gemini for GRACE 2.0."},
    {"text": "What LLM experience does Sidharth have?", "output": "He has worked with OpenAI models, Google Gemini, and HuggingFace. He has built AI-powered browser extensions and chatbot features."},
    {"text": "Does Sidharth know Langchain?", "output": "Yes, Langchain is listed in his AI/ML skills. It's a framework for building applications with large language models."},
    {"text": "Has Sidharth worked with OpenAI?", "output": "Yes, he integrated OpenAI models into GRACE Sync for AI chatbot functionality and various AI features."},
    {"text": "Does Sidharth know about machine learning?", "output": "Yes, at TCS Rapid Labs he built ML models including an ulcer detection model with 92% precision and a ticket prediction model with 85% accuracy."},
    {"text": "What is Sidharth's AI experience?", "output": "He has built AI-powered browser extensions, integrated OpenAI and Gemini models, worked with Langchain and HuggingFace, and deployed ML models for healthcare applications."},
    
    # Databases
    {"text": "What database systems is he familiar with?", "output": "He uses MongoDB and CosmosDB."},
    {"text": "Does Sidharth know MongoDB?", "output": "Yes, MongoDB is one of his primary databases. He used it in the MERN stack for Project Convergence and implemented automated backup systems."},
    {"text": "What NoSQL databases does Sidharth use?", "output": "He uses MongoDB and Azure Cosmos DB for NoSQL database needs."},
    {"text": "Does Sidharth know SQL?", "output": "His main database experience is with NoSQL databases like MongoDB and Cosmos DB. He also worked with DynamoDB on AWS."},
    {"text": "What databases has Sidharth worked with?", "output": "He has experience with MongoDB, Azure Cosmos DB, and AWS DynamoDB."},
    
    # Tools & Development
    {"text": "What development tools does Sidharth use?", "output": "He utilizes Git, Postman, VS Code, Cursor, Windsurf, Antigravity, and Microsoft Office tools like Excel and PowerPoint."},
    {"text": "Does Sidharth use Git?", "output": "Yes, Git is one of his primary tools for version control."},
    {"text": "What IDE does Sidharth use?", "output": "He uses VS Code as his primary IDE, along with AI-assisted tools like Cursor, Windsurf, and Antigravity."},
    {"text": "Does Sidharth know Postman?", "output": "Yes, Postman is listed in his tools. He uses it for API testing and development."},
    {"text": "What AI coding tools does Sidharth use?", "output": "He uses AI-assisted development tools including Cursor, Windsurf, and Antigravity for enhanced coding productivity."},
    
    # Contact Information
    {"text": "How can I contact Sidharth E?", "output": "You can email him at sidharthe38943@gmail.com. His portfolio is available at https://sidharthe.dev."},
    {"text": "What is Sidharth's email?", "output": "His email address is sidharthe38943@gmail.com."},
    {"text": "Does Sidharth have a portfolio website?", "output": "Yes, his portfolio is available at https://sidharthe.dev."},
    {"text": "Where can I see Sidharth's work?", "output": "You can visit his portfolio website at https://sidharthe.dev to see his projects and experience."},
    {"text": "How do I reach Sidharth?", "output": "You can reach Sidharth via email at sidharthe38943@gmail.com, or visit his portfolio at https://sidharthe.dev."},
    
    # KellyOCG Projects
    {"text": "Tell me about his work at KellyOCG.", "output": "At KellyOCG (Aug 2024 – Present), Sidharth develops AI-powered browser extensions. Key projects include 'GRACE Sync' and 'Project GRACE 2.0', focusing on Azure, OpenAI, and Cosmos DB integrations."},
    {"text": "What is GRACE Sync?", "output": "GRACE Sync is an AI-powered browser extension developed by Sidharth. It processes data from internal web apps, features an AI chatbot, and uses Azure Identity for security. It also includes telemetry via PowerBI and history management with Cosmos DB."},
    {"text": "What features did he implement in Project GRACE 2.0?", "output": "For GRACE 2.0, he integrated file uploads with Gemini, added OCR for images, implemented real-time web search via Perplexity API, ensured Veracode compliance, and added Role-Based Access Control (RBAC)."},
    {"text": "What is Project GRACE?", "output": "GRACE is an AI-powered browser extension project at KellyOCG. It includes GRACE Sync (AI chatbot with data processing) and GRACE 2.0 (enhanced with Gemini, OCR, web search, and security features)."},
    {"text": "What technologies were used in GRACE Sync?", "output": "GRACE Sync uses Azure Identity for security, OpenAI for AI chatbot functionality, Cosmos DB for history management, PowerBI for telemetry, and Bicep for infrastructure."},
    {"text": "What is the OCR feature in GRACE 2.0?", "output": "In GRACE 2.0, Sidharth added OCR (Optical Character Recognition) capabilities for processing images, integrated with the Gemini AI model for enhanced functionality."},
    
    # Poumki Projects
    {"text": "What did Sidharth do at Poumki Digital LLP?", "output": "From April 2023 to June 2024, he worked as a Full Stack Developer. He built the 'Project Convergence' data analytics platform, the 'e-slide' presentation platform, and enhanced the 'Izycourse' platform."},
    {"text": "Describe Project Convergence.", "output": "Project Convergence is a Data Analytics Platform built on the MERN stack. Sidharth implemented CI/CD with Jenkins on AWS, set up a rollback system, reduced storage costs with Nexus/S3, and automated MongoDB backups."},
    {"text": "What is Project e-slide?", "output": "It is a presentation sharing platform where Sidharth added multilingual support using React-i18next and implemented user safety features like banning capabilities. The tech stack included React, SASS, DynamoDB, and AWS Lambda."},
    {"text": "What is Izycourse?", "output": "Izycourse is an e-learning platform that Sidharth enhanced during his time at Poumki Digital LLP, including UI improvements and feature enhancements."},
    {"text": "What technologies were used in Project Convergence?", "output": "Project Convergence was built on the MERN stack (MongoDB, ExpressJS, React, Node.js) with Jenkins for CI/CD, AWS for hosting, Nexus and S3 for storage, and Socket.IO for real-time features."},
    {"text": "What is the MERN stack?", "output": "MERN stack consists of MongoDB, ExpressJS, React, and Node.js. Sidharth used this stack to build Project Convergence at Poumki Digital LLP."},
    
    # TCS Experience
    {"text": "What was Sidharth's role at TCS?", "output": "He held two roles at TCS: Site Reliability Engineer (Jan 2020 – Mar 2021) and Full Stack Developer for Rapid Labs (Mar 2021 – Apr 2023)."},
    {"text": "Did Sidharth work on healthcare projects?", "output": "Yes, at TCS Rapid Labs, he built AI/ML prototypes for healthcare, including an ulcer detection model with 92% precision and a ticket prediction model with 85% accuracy using Python and Flask."},
    {"text": "What automation tools did he build as an SRE?", "output": "He built a 'Core Analysis Tool' to reduce incident resolution time by 30%, an 'Akamai report generator' saving 200+ hours annually, and an 'Auto Delegation Tool' to validate data discrepancies."},
    {"text": "What is TCS Rapid Labs?", "output": "TCS Rapid Labs is an innovation hub at Tata Consultancy Services where Sidharth worked as a Full Stack Developer from March 2021 to April 2023, building AI/ML prototypes for healthcare."},
    {"text": "What ML models did Sidharth build?", "output": "At TCS Rapid Labs, he built an ulcer detection model with 92% precision and a ticket prediction model with 85% accuracy using Python, Flask, and various ML frameworks."},
    {"text": "What was the Core Analysis Tool?", "output": "The Core Analysis Tool was an automation tool Sidharth built at TCS that reduced incident resolution time by 30%."},
    {"text": "What was Sidharth's SRE experience?", "output": "As a Site Reliability Engineer at TCS (Jan 2020 – Mar 2021), Sidharth built automation tools including the Core Analysis Tool, Akamai report generator, and Auto Delegation Tool, saving over 500 work hours annually."},
    
    # Education
    {"text": "What is Sidharth's educational background?", "output": "He is currently pursuing a Master of Computer Applications (MCA) in AI/ML at Amritha Vishwa Vidyapeetham (since Oct 2025). He holds a Bachelor of Computer Applications (BCA) from VLB Janakiammal College of Arts and Science (2017–2020)."},
    {"text": "Does Sidharth have a degree?", "output": "Yes, he has a Bachelor of Computer Applications (BCA) from VLB Janakiammal College of Arts and Science (2017–2020) and is pursuing an MCA in AI/ML at Amritha Vishwa Vidyapeetham."},
    {"text": "Where did Sidharth study?", "output": "He completed his BCA at VLB Janakiammal College of Arts and Science (2017-2020) and is currently pursuing MCA in AI/ML at Amritha Vishwa Vidyapeetham."},
    {"text": "What is Sidharth studying now?", "output": "He is currently pursuing a Master of Computer Applications (MCA) with specialization in AI/ML at Amritha Vishwa Vidyapeetham."},
    {"text": "Is Sidharth a graduate?", "output": "Yes, he has a Bachelor of Computer Applications (BCA) degree and is currently pursuing his Master's degree (MCA) in AI/ML."},
    
    # Awards & Recognition
    {"text": "Has Sidharth received any awards?", "output": "Yes, he received the 'Star of the Month' award at TCS for developing three automation tools that saved over 500 work hours annually."},
    {"text": "What recognition has Sidharth received?", "output": "He received the 'Star of the Month' award at TCS for his automation tools that significantly improved operational efficiency."},
    {"text": "Did Sidharth get any awards at TCS?", "output": "Yes, he received the 'Star of the Month' award for developing three automation tools (Core Analysis Tool, Akamai report generator, Auto Delegation Tool) that saved over 500 work hours annually."},
    
    # Languages
    {"text": "What languages does Sidharth speak?", "output": "He speaks Malayalam (Native), English (Professional Proficiency), Tamil (Conversational), and Hindi (Conversational)."},
    {"text": "Is Sidharth fluent in English?", "output": "Yes, he has professional proficiency in English. It is one of his primary languages along with Malayalam."},
    {"text": "What is Sidharth's native language?", "output": "His native language is Malayalam. He is from Kerala, India."},
    {"text": "Can Sidharth speak Hindi?", "output": "Yes, he has conversational proficiency in Hindi."},
    
    # Technical Details
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
    {"text": "Does Sidharth do UI/UX design?", "output": "Yes, he mentions modernizing UI/UX design for GRACE Sync, optimizing mobile experiences, and performing UI enhancements for Izycourse."},
    
    # Additional Skill Questions
    {"text": "Does Sidharth know JWT?", "output": "Yes, he has experience implementing JWT (JSON Web Tokens) for authentication in REST APIs."},
    {"text": "Can Sidharth work with WebSockets?", "output": "Yes, WebSockets is listed under his 'Other' skills, and he has used Socket.IO for real-time features in Project Convergence."},
    {"text": "Does Sidharth know about Nginx?", "output": "Yes, Nginx is part of his DevOps toolkit. He uses it as a web server and reverse proxy."},
    {"text": "What is MCP in Sidharth's AI skills?", "output": "MCP (Model Context Protocol) is listed in his AI/ML skills, referring to a standard for connecting AI systems with external tools and data sources."},
    {"text": "Does Sidharth have Agentic AI experience?", "output": "Yes, Agentic AI is listed in his AI/ML skills, indicating experience with autonomous AI systems that can take actions."},
    
    # Project-specific questions
    {"text": "What security measures did Sidharth implement in GRACE 2.0?", "output": "He achieved full Veracode compliance by fixing high-severity vulnerabilities and implemented Role-Based Access Control (RBAC) for enhanced security."},
    {"text": "What storage solutions did Sidharth use?", "output": "He has used AWS S3, Nexus repository, MongoDB, Cosmos DB, and DynamoDB for various storage needs across projects."},
    {"text": "Has Sidharth worked with PowerBI?", "output": "Yes, he integrated PowerBI for telemetry and analytics in GRACE Sync at KellyOCG."},
    {"text": "What multilingual features has Sidharth implemented?", "output": "He added multilingual support to Project e-slide using React-i18next for internationalization."},
    {"text": "Did Sidharth implement any rollback systems?", "output": "Yes, he set up a rollback system for Project Convergence to ensure safe deployments and easy recovery from issues."},
    
    # Casual/Conversational Questions
    {"text": "Hello", "output": "Hello! I'm here to answer questions about Sidharth E. Feel free to ask about his skills, experience, projects, or contact information!"},
    {"text": "Hi", "output": "Hi there! How can I help you learn more about Sidharth E today?"},
    {"text": "Hey", "output": "Hey! I'm an AI assistant for Sidharth E's portfolio. What would you like to know about him?"},
    {"text": "What can you tell me?", "output": "I can tell you about Sidharth E's professional experience, technical skills, projects, education, and contact information. What would you like to know?"},
    {"text": "Help", "output": "I can help you learn about Sidharth E. Try asking about his skills (React, Node.js, AI), experience (KellyOCG, TCS, Poumki), projects (GRACE, Convergence), or contact details."},
    {"text": "Thanks", "output": "You're welcome! Feel free to ask if you have more questions about Sidharth E."},
    {"text": "Thank you", "output": "You're welcome! Let me know if there's anything else you'd like to know about Sidharth E."},
    {"text": "Bye", "output": "Goodbye! Feel free to come back if you have more questions about Sidharth E."},
    {"text": "Goodbye", "output": "Goodbye! Thanks for learning about Sidharth E. Visit his portfolio at https://sidharthe.dev for more information."},
    
    # Edge cases and variations
    {"text": "Is Sidharth available for hire?", "output": "Sidharth is currently working as a Full Stack Developer at KellyOCG. For job inquiries or collaboration opportunities, you can contact him at sidharthe38943@gmail.com."},
    {"text": "Can Sidharth work remotely?", "output": "For work arrangement inquiries, please contact Sidharth directly at sidharthe38943@gmail.com."},
    {"text": "What is Sidharth's specialty?", "output": "Sidharth specializes in Full Stack Development with expertise in React, Node.js, and AI integrations. He excels at building AI-powered applications and browser extensions."},
    {"text": "What makes Sidharth unique?", "output": "Sidharth combines Full Stack Development skills with AI/ML expertise. He has built production AI chatbots, ML models for healthcare, and award-winning automation tools that saved 500+ work hours."},
    {"text": "Why should I hire Sidharth?", "output": "Sidharth brings nearly 5 years of experience, strong Full Stack skills (React, Node.js, Azure, AWS), AI integration expertise, and a track record of building impactful tools. His automation work at TCS saved 500+ hours annually."},
    {"text": "What is Sidharth's strongest skill?", "output": "Sidharth excels at Full Stack Development with a strong focus on AI integrations. He combines frontend expertise (React, Next.js) with backend skills (Node.js, Python) and modern AI tools (OpenAI, Gemini, Langchain)."},
    {"text": "How experienced is Sidharth with AI?", "output": "Sidharth has significant AI experience. He has built AI-powered browser extensions with OpenAI, integrated Gemini for file processing, created ML models for healthcare at TCS, and works with Langchain and HuggingFace."},
    
    # Additional Skills - Frontend Details
    {"text": "What is Sidharth's experience with HTML and CSS?", "output": "Sidharth has strong expertise in HTML5/CSS/JS, which forms the foundation of his frontend development skills. He uses these along with modern frameworks like React and Next.js."},
    {"text": "Does Sidharth know SASS?", "output": "Yes, SASS is one of his CSS tools. He uses it alongside Tailwind CSS for styling web applications."},
    {"text": "Can Sidharth build modern web apps?", "output": "Absolutely! He builds modern web applications using React, Next.js, TypeScript, and styling with Tailwind CSS and SASS. His apps include AI integrations and real-time features."},
    
    # Additional Backend Skills
    {"text": "What is FastAPI?", "output": "FastAPI is a modern Python web framework that Sidharth uses for building high-performance APIs. It's listed in his backend skills alongside Node.js and ExpressJS."},
    {"text": "Does Sidharth know Django?", "output": "Yes, he used Django for building web applications at TCS Rapid Labs, particularly for healthcare AI/ML prototypes."},
    {"text": "What Python frameworks does Sidharth know?", "output": "Sidharth knows FastAPI, Flask, and Django for Python web development. He has used these for building APIs and ML-powered applications."},
    
    # Additional Database Skills
    {"text": "Does Sidharth know DynamoDB?", "output": "Yes, he used AWS DynamoDB for Project e-slide, a presentation sharing platform built with React, SASS, and AWS Lambda."},
    {"text": "Does Sidharth know PostgreSQL?", "output": "Yes, PostgreSQL was used in Project e-slide alongside DynamoDB, AWS Lambda, and GraphQL."},
    {"text": "Does Sidharth know MySQL?", "output": "Yes, he used MySQL with .NET and C# for building SRE automation tools at TCS."},
    {"text": "What is Cosmos DB?", "output": "Azure Cosmos DB is a NoSQL database that Sidharth uses. He implemented chat history management with Cosmos DB for GRACE Sync at KellyOCG."},
    {"text": "Has Sidharth automated database backups?", "output": "Yes, he created scripts to automate MongoDB backups and used AWS Lambda with S3 to automate database backups for Project Convergence."},
    
    # Additional Cloud Skills
    {"text": "Does Sidharth know Bicep?", "output": "Yes, Bicep is listed in his Cloud/DevOps skills. He uses it for Azure infrastructure as code, particularly for GRACE 2.0."},
    {"text": "What AWS services has Sidharth used?", "output": "He has used AWS EC2, S3, Lambda, DynamoDB, and Amplify across various projects including Convergence and e-slide."},
    {"text": "Does Sidharth know AWS Amplify?", "output": "Yes, he used AWS Amplify along with GraphQL for Project e-slide, a presentation sharing platform."},
    {"text": "What is Sidharth's experience with VPCs?", "output": "He designed secure VPCs based on AWS best practices for Project Convergence at Poumki Digital LLP."},
    {"text": "Does Sidharth know about monitoring and logging?", "output": "Yes, he has implemented centralized logging using Azure Monitor for GRACE 2.0 and used Winston and LogLevel logging for full observability in Project Convergence."},
    
    # Additional AI/ML Skills
    {"text": "What is HuggingFace?", "output": "HuggingFace is an AI/ML platform that Sidharth uses. It provides access to thousands of pre-trained models and is part of his AI toolkit."},
    {"text": "What is Langchain?", "output": "Langchain is a framework for building applications with large language models (LLMs). Sidharth lists it in his AI/ML skills."},
    {"text": "Does Sidharth know Gemini?", "output": "Yes, he integrated Google Gemini in GRACE 2.0 for file uploads and document processing capabilities."},
    {"text": "What is Agentic AI?", "output": "Agentic AI refers to autonomous AI systems that can take actions. Sidharth lists it in his AI/ML skills, indicating experience with AI agents."},
    {"text": "Does Sidharth know Azure AI?", "output": "Yes, Azure AI is part of his AI/ML skills. He uses various Azure AI services in his projects at KellyOCG."},
    {"text": "What is the Perplexity API?", "output": "Sidharth implemented real-time web search using the Perplexity API in GRACE 2.0, enhancing the chatbot's response relevance."},
    
    # Additional Tools
    {"text": "What is Cursor?", "output": "Cursor is an AI-assisted code editor that Sidharth uses. It's listed under his development tools alongside VS Code and Windsurf."},
    {"text": "What is Windsurf?", "output": "Windsurf is an AI-assisted development tool that Sidharth uses for enhanced coding productivity."},
    {"text": "Does Sidharth know Excel?", "output": "Yes, Microsoft Excel is listed in his tools. He uses it along with PowerPoint and Word for documentation and reporting."},
    {"text": "What Microsoft Office tools does Sidharth use?", "output": "He uses Excel, PowerPoint, and Word for documentation, presentations, and data analysis."},
    
    # Other Skills
    {"text": "Does Sidharth know GraphQL?", "output": "Yes, he used GraphQL with AWS Amplify for Project e-slide, a presentation sharing platform."},
    {"text": "What is Socket.IO?", "output": "Socket.IO is a library for real-time communication. Sidharth used it to implement real-time user interactions in Project Convergence."},
    
    # KellyOCG Project Details
    {"text": "What telemetry features did Sidharth implement?", "output": "He implemented Telemetry collection for PowerBI dashboards in GRACE Sync to monitor extension usage and performance metrics."},
    {"text": "How did Sidharth optimize mobile experience?", "output": "In GRACE 2.0, he optimized mobile experience with dynamic textbox resizing, achieving 95% user satisfaction."},
    {"text": "What is the prompt saving feature?", "output": "In GRACE 2.0, Sidharth implemented new prompt saving functionality in the UI, enabling users to easily store and reuse custom prompts."},
    {"text": "Did Sidharth implement clipboard paste for images?", "output": "Yes, he implemented functionality to paste images directly from clipboard in GRACE 2.0 for improved user experience."},
    {"text": "What codeblock improvements did Sidharth make?", "output": "He redesigned the codeblock component in GRACE 2.0 to enhance readability and user interaction with code snippets."},
    {"text": "What security certifications did GRACE 2.0 achieve?", "output": "Sidharth achieved full Veracode compliance for GRACE 2.0 by fixing all high-severity vulnerabilities and implementing RBAC."},
    {"text": "Does Sidharth handle penetration test feedback?", "output": "Yes, he led monthly UI/security updates based on penetration test feedback for GRACE 2.0."},
    
    # Poumki Project Details
    {"text": "How did Sidharth reduce storage costs?", "output": "He connected Nexus with AWS S3 to reduce artifact storage costs by 30% for Project Convergence."},
    {"text": "What test coverage did Project Convergence have?", "output": "Sidharth built a multilingual UI with 100% test coverage for Project Convergence."},
    {"text": "What pagination features did Sidharth implement?", "output": "He implemented pagination and search functionality with frontend filters in Project Convergence to enhance user experience and data accessibility."},
    {"text": "What user safety features did Sidharth add?", "output": "He implemented a ban user feature in Project e-slide to improve community safety."},
    {"text": "What is React-i18next?", "output": "React-i18next is an internationalization framework. Sidharth used it to add multilingual support to Project e-slide and Izycourse."},
    
    # TCS Project Details
    {"text": "What was Sidharth's ulcer detection model accuracy?", "output": "The ulcer detection prototype model he developed at TCS Rapid Labs achieved 92% precision."},
    {"text": "What was the ticket prediction model accuracy?", "output": "The ticket prediction prototype model he built at TCS Rapid Labs achieved 85% accuracy."},
    {"text": "Did Sidharth do any migration work?", "output": "Yes, he migrated an internally developed Django app to AngularJS at TCS Rapid Labs."},
    {"text": "How much time did Sidharth's automation tools save?", "output": "His automation tools at TCS saved over 500 work hours annually, which earned him the 'Star of the Month' award."},
    {"text": "What was the Auto Delegation Tool?", "output": "The Auto Delegation Tool was an automation tool Sidharth built at TCS to validate data discrepancies with different data sources, reducing manual effort by 40%."},
    {"text": "How much time did the Core Analysis Tool save?", "output": "The Core Analysis Tool reduced incident resolution time by 30% at TCS by enabling root cause analysis for service incidents."},
    
    # Education Details
    {"text": "What is Sidharth's MCA specialization?", "output": "His MCA is specialized in AI/ML (Artificial Intelligence and Machine Learning) at Amritha Vishwa Vidyapeetham."},
    {"text": "Is Sidharth pursuing higher education?", "output": "Yes, he is currently pursuing a Master of Computer Applications (MCA) with AI/ML specialization at Amritha Vishwa Vidyapeetham."},
    {"text": "When did Sidharth complete his BCA?", "output": "He completed his Bachelor of Computer Applications (BCA) from VLB Janakiammal College of Arts and Science, Coimbatore in September 2020."},
    
    # Language Details
    {"text": "Does Sidharth speak Tamil?", "output": "Yes, he has conversational proficiency in Tamil."},
    {"text": "How many languages does Sidharth speak?", "output": "He speaks four languages: Malayalam (Native), English (Professional Proficiency), Tamil (Conversational), and Hindi (Conversational)."},
    {"text": "Is Sidharth from Kerala?", "output": "Yes, Sidharth E is from Kerala, India. Malayalam is his native language."},
    
    # Summary Questions
    {"text": "Summarize Sidharth's skills", "output": "Sidharth's skills span: Frontend (React, Next.js, TypeScript, Tailwind), Backend (Node.js, Python, FastAPI), Databases (MongoDB, Cosmos DB), Cloud (Azure, AWS, Jenkins), AI/ML (OpenAI, Gemini, Langchain, HuggingFace), and Tools (Git, VS Code, Cursor)."},
    {"text": "What is Sidharth's tech stack?", "output": "His primary stack includes React/Next.js for frontend, Node.js/Python for backend, MongoDB/Cosmos DB for databases, Azure/AWS for cloud, and OpenAI/Gemini for AI. He also uses TypeScript, Tailwind CSS, and modern DevOps tools."},
    {"text": "List all of Sidharth's frontend skills", "output": "Frontend: HTML5/CSS/JS, TypeScript, React, Next.js, SASS, Tailwind CSS, AngularJS, Vue.js."},
    {"text": "List all of Sidharth's backend skills", "output": "Backend: Node.js, ExpressJS, Python, FastAPI, Flask, Django."},
    {"text": "List all of Sidharth's database skills", "output": "Databases: MongoDB, Azure Cosmos DB, AWS DynamoDB, PostgreSQL, MySQL."},
    {"text": "List all of Sidharth's cloud skills", "output": "Cloud/DevOps: AWS, Azure, Jenkins, Nginx, Bicep, Linux, CI/CD Pipelines, AWS Amplify."},
    {"text": "List all of Sidharth's AI skills", "output": "AI/ML: Agentic AI, Langchain, HuggingFace, MCP, GenAI, OpenAI, Google Gemini, Azure AI."},
    {"text": "What projects has Sidharth worked on?", "output": "Key projects: GRACE Sync (AI browser extension), GRACE 2.0 (GenAI chat app), Project Convergence (data analytics), e-slide (presentations), Izycourse (e-learning), and healthcare AI prototypes at TCS."},
    
    # More Conversational
    {"text": "Can you recommend Sidharth for a job?", "output": "Absolutely! Sidharth brings nearly 5 years of Full Stack experience, strong AI integration skills, and a track record of building impactful products. He's worked at TCS, Poumki Digital, and currently KellyOCG. Contact him at sidharthe38943@gmail.com."},
    {"text": "What are Sidharth's career highlights?", "output": "Key highlights: Built AI browser extensions at KellyOCG, created ML models with 92% precision at TCS, implemented CI/CD saving 30% storage costs, won 'Star of the Month' for automation tools saving 500+ hours annually."},
    {"text": "Tell me about Sidharth's achievements", "output": "Notable achievements: 'Star of the Month' award at TCS, 92% precision ML model, 95% user satisfaction for mobile UX, 100% test coverage for multilingual apps, full Veracode security compliance."},
    {"text": "What is Sidharth best known for?", "output": "Sidharth is known for building AI-powered applications, including browser extensions with chatbots, and creating automation tools that save hundreds of work hours. He combines Full Stack skills with modern AI technologies."},
    {"text": "Describe Sidharth in one sentence", "output": "Sidharth E is a Full Stack Developer from Kerala, India with nearly 5 years of experience specializing in React, Node.js, cloud platforms, and AI integrations."}
]

# Convert to Alpaca format (instruction + response) with EOS token
# The <|end▁of▁sentence|> token tells the model when to STOP generating
alpaca_prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
{}<|end▁of▁sentence|>"""

formatted_data = [{"text": alpaca_prompt.format(item["text"], item["output"])} for item in raw_data]
dataset = Dataset.from_list(formatted_data)
print(f"Training on {len(formatted_data)} examples")

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
            num_train_epochs=15,  # More epochs for small dataset
            learning_rate=2e-4,   # TinyLlama can handle higher LR
            warmup_steps=10,      # Warmup for stable training
            fp16=True,
            logging_steps=5,
            optim="paged_adamw_8bit",
            save_strategy="no",
            weight_decay=0.01,    # Prevent overfitting
        ),
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    
    trainer.train()
    
    print("Saving adapter...")
    model.save_pretrained("Sidharth_AI_Model")
    print("Done!")
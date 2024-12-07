# World-Building-AI-Assistant  

World-Building-AI-Assistant is an AI-powered project designed to assist storytellers, game designers, and writers in creating immersive, rich, and detailed worlds. This repository focuses on efficiently fine-tuning large language models, like GPT-Neo, using a structured dataset that represents various elements of world-building, such as civilizations, factions, regions, and historical events.  

---

## 📖 Overview  
This repository is a comprehensive toolkit for training language models to understand and generate intricate world-building details. By utilizing **Low-Rank Adaptation (LoRA)** and **8-bit quantization**, we achieve resource-efficient model fine-tuning on a small but representative dataset. The goal is to empower creators with AI-generated narratives that are both high-quality and tailored to their creative needs.  

---

## 🌟 Introduction  
Storytelling is at the heart of human creativity, but building a believable, complex, and immersive world can be a daunting task. With limited resources, we sought to simplify this process by leveraging AI to create dynamic, detailed content.  

The **World-Building-AI-Assistant** serves as a foundational tool for:  
- **Game Designers**: Generate civilizations, factions, and quests for RPGs and world-building games.  
- **Storytellers**: Create detailed histories, societies, and regions for novels, screenplays, and campaigns.  
- **Hobbyists**: Explore your imagination with AI-assisted tools for your creative endeavors.  

Our small, carefully designed dataset ensures meaningful insights while maintaining computational efficiency, making this project accessible even on systems with limited hardware capabilities.  

---

## 🎯 Vision  
The vision of this project is to democratize AI-powered world-building by enabling creators to train and use advanced language models without requiring significant computational resources. By fine-tuning a small yet comprehensive dataset, we aim to deliver a scalable framework for efficient storytelling.  

---

## 🔬 Methodology  

### 1. **Dataset Details**  
The dataset is a structured JSON file comprising:  
- **World States**: Current state descriptions, societal issues, technological levels, and major conflicts.  
- **Regions**: Geographies, climates, terrains, natural resources, and environmental hazards.  
- **Factions**: Motivations, resources, notable members, allies, and enemies.  
- **Historical Background**: Key events, civilizations, artifacts, and their influence on the present world.  

Due to resource constraints, we utilized a **small dataset** to represent diverse aspects of world-building. This dataset prioritizes quality and structure, ensuring that the AI learns effectively despite its limited size.

### 2. **Model Fine-Tuning**  
We used the **EleutherAI GPT-Neo 2.7B** model, fine-tuned with the following techniques:  
- **8-bit Quantization**: Reduces memory usage and improves performance on consumer-grade GPUs.  
- **Low-Rank Adaptation (LoRA)**: Efficiently updates small parameters, avoiding the need for full model fine-tuning.  
- **Tokenization**: Ensures optimal input formatting for training and inference.  

### 3. **Training Pipeline**  
1. **Data Preprocessing**: Converts the JSON dataset into descriptive text inputs.  
2. **Tokenization**: Processes data into a model-readable format with truncation and padding.  
3. **Model Preparation**: Loads GPT-Neo with memory optimization using `bitsandbytes`.  
4. **LoRA Integration**: Targets attention modules (`k_proj`, `v_proj`, and `q_proj`) for fine-tuning.  
5. **Training**: Employs **gradient accumulation** and **low learning rates** for stable training on small datasets.

---

## 🔮 Future Work  

We plan to expand the project to:  
- **Incorporate Larger Datasets**: Collect more diverse and detailed data for even richer outputs.  
- **Support More Models**: Add compatibility with cutting-edge models like GPT-4 or Falcon.  
- **Interactive Features**: Enable real-time world-building suggestions through an API or web-based interface.  
- **Integration with Game Engines**: Export AI-generated content to tools like Unreal Engine or Unity for seamless game development.  
- **Multilingual Support**: Train the AI to generate world-building content in multiple languages.  

---

## 🤝 Contributing  

Contributions are welcome! Whether you want to improve the dataset, optimize the model training pipeline, or propose new features, feel free to fork the repository and submit a pull request.  

---

## 📧 Contact  

If you have any questions, feedback, or suggestions, feel free to reach out:  
- **GitHub Issues**: [Open an Issue](https://github.com/yourusername/World-Building-AI-Assistant/issues)  
- **Email**: sourabh.meena4444@gmail.com
- **LinkedIn: www.linkedin.com/in/sourabh-meena-83749a164

---

---


#  Install required packages
!pip install -q transformers==4.36.2 datasets==2.17.0 accelerate==0.27.2 bitsandbytes==0.41.3 peft==0.8.2

# Import necessary libraries
import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
from google.colab import files
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import gc
import os

# Configure environment for stability
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cuda.matmul.allow_tf32 = True

# Function to prepare world-building text data
def prepare_text(json_data):
    texts = []

    for entry in json_data:
        if 'world_building' in entry:
            world_state = entry['world_building']['world_state']

            # World description
            world_desc = f"World Description: {world_state['current_state'].get('description', 'No description provided.')}\n\n"

            # Major conflicts
            conflicts = "Major Conflicts:\n" + "\n".join(f"- {conflict}"
                for conflict in world_state['current_state'].get('major_conflicts', [])) + "\n\n"

            # Power structures
            powers = "Power Structures:\n" + "\n".join(f"- {power}"
                for power in world_state['current_state'].get('power_structures', [])) + "\n\n"

            # Societal issues
            societal_issues = "Societal Issues:\n" + "\n".join(f"- {issue}"
                for issue in world_state['current_state'].get('societal_issues', [])) + "\n\n"

            # Technological level
            tech_level = f"Technological Level: {world_state['current_state'].get('technological_level', 'No data')}\n\n"

            # Historical events
            history = "Historical Events:\n"
            for event in world_state['historical_background'].get('major_events', []):
                history += f"- {event.get('event', 'No event name')} ({event.get('date', 'No date')}): {event.get('impact', 'No impact description')}.\n"
                if 'remaining_evidence' in event:
                    history += f"  Remaining Evidence: {', '.join(event['remaining_evidence'])}\n"
            history += "\n"

            # Civilizations
            civilizations = "Civilizations:\n"
            for civ in world_state['historical_background'].get('civilizations', []):
                civilizations += f"- {civ.get('name', 'No name')}: {civ.get('legacy', 'No legacy description')}.\n"
                civilizations += f"  Artifacts: {', '.join(civ.get('artifacts', []))}\n"
                civilizations += f"  Influence on Present: {civ.get('influence_on_present', 'No influence description')}.\n"
            civilizations += "\n"

            # Combine all world-building data
            full_text = world_desc + conflicts + powers + societal_issues + tech_level + history + civilizations
            texts.append(full_text)

            # Regions
            for region in entry['world_building'].get('regions', []):
                region_text = f"Region: {region.get('name', 'No region name')}\n"
                geography = region.get('geography', {})
                region_text += f"Climate: {geography.get('climate', 'No climate data')}\n"
                region_text += f"Terrain: {geography.get('terrain', 'No terrain data')}\n"
                region_text += f"Natural Barriers: {', '.join(geography.get('natural_barriers', []))}\n"
                region_text += f"Landscape: {geography.get('landscape', 'No landscape description')}\n"
                region_text += f"Notable Features: {', '.join(geography.get('notable_features', []))}\n"
                region_text += f"Environmental Hazards: {', '.join(geography.get('environmental_hazards', []))}\n"
                resource_dist = geography.get('resource_distribution', {})
                region_text += f"Resource Distribution:\n"
                region_text += f"  Type: {resource_dist.get('type', 'No type specified')}\n"
                region_text += f"  Location: {resource_dist.get('location', 'No location specified')}\n"
                region_text += f"  Scarcity Level: {resource_dist.get('scarcity_level', 'No scarcity info')}\n\n"

                # Locations
                for location in region.get('locations', []):
                    region_text += f"Location: {location.get('name', 'No location name')}\n"
                    description = location.get('description', {})
                    region_text += f"  Description: {description.get('visual_elements', 'No visual elements')}\n"
                    region_text += f"  Atmosphere: {description.get('atmosphere', 'No atmosphere description')}\n"
                    region_text += f"  Unique Features: {description.get('unique_features', 'No unique features')}\n"
                    region_text += f"  Daily Life: {description.get('daily_life', 'No daily life description')}\n\n"

                    # Points of interest
                    for poi in location.get('points_of_interest', []):
                        region_text += f"  Point of Interest: {poi.get('name', 'No POI name')}\n"
                        region_text += f"    Description: {poi.get('description', 'No POI description')}\n"
                        region_text += f"    Gameplay Purpose: {poi.get('gameplay_purpose', 'No gameplay purpose')}\n"
                        region_text += f"    Story Significance: {poi.get('story_significance', 'No story significance')}\n"
                        region_text += f"    Hidden Elements: {', '.join(poi.get('hidden_elements', []))}\n\n"

                texts.append(region_text)

            # Factions
            if 'factions' in entry['world_building']:
                faction_text = "Factions:\n"
                for faction in entry['world_building']['factions']:
                    faction_text += f"Faction Name: {faction.get('name', 'No faction name')}\n"
                    faction_text += f"Description: {faction.get('description', 'No description')}\n"
                    faction_text += f"Motives: {', '.join(faction.get('motives', []))}\n"
                    faction_text += f"Resources: {', '.join(faction.get('resources', []))}\n"
                    faction_text += f"Allies: {', '.join(faction.get('allies', []))}\n"
                    faction_text += f"Enemies: {', '.join(faction.get('enemies', []))}\n"

                    # Notable members
                    if 'notable_members' in faction:
                        faction_text += "Notable Members:\n"
                        for member in faction['notable_members']:
                            faction_text += f"  - Name: {member.get('name', 'No name')}\n"
                            faction_text += f"    Role: {member.get('role', 'No role')}\n"
                            faction_text += f"    Background: {member.get('background', 'No background')}\n"
                    faction_text += "\n"

                texts.append(faction_text)

    return texts

# Load and process data
print("Please upload your world.json file")
uploaded = files.upload()

with open('world.json', 'r') as file:
    json_data = json.load(file)

training_texts = prepare_text(json_data)
dataset = Dataset.from_dict({'text': training_texts})

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    'EleutherAI/gpt-neo-2.7B',
    model_max_length=512,
    padding_side='right'
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=512,
        return_tensors=None
    )

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=['text']
)

# Initialize model with 8-bit quantization and LoRA
print("Loading model... This may take a few minutes...")
model = AutoModelForCausalLM.from_pretrained(
    'EleutherAI/gpt-neo-2.7B',
    device_map='auto',
    load_in_8bit=True,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)

# Prepare model for k-bit training
model = prepare_model_for_kbit_training(model)

# Configure LoRA with available target modules
lora_config = LoraConfig(
    r=16,                     # Rank
    lora_alpha=32,           # Alpha scaling
    target_modules=["k_proj", "v_proj", "q_proj"],  # Target attention modules
    lora_dropout=0.05,       # Dropout probability
    bias="none",
    task_type="CAUSAL_LM"    # Task type for causal language modeling
)


# Create PEFT model
model = get_peft_model(model, lora_config)
print("Number of trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

# Set up training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-4,  # Slightly higher learning rate for LoRA
    fp16=True,
    save_steps=100,
    logging_steps=10,
    save_total_limit=2,
    evaluation_strategy='no',
    report_to='none',
    remove_unused_columns=False,  # Important for avoiding issues with the dataset
    dataloader_pin_memory=False,
)

# Set up trainer and start training
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Clear memory before training
gc.collect()
torch.cuda.empty_cache()

print("Setting up trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

print("Starting training...")
trainer.train()

# Save the final model
print("Saving model...")
model.save_pretrained('./final_model')
print("model saved")

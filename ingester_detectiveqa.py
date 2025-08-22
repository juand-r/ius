"""
First run python download_detectiveqa_novels.py to download the novels into data-source/detectiveqa_novels/\
Each novel is in a .txt file labels like 87-<...>.txt, i.e., the first number is the book id. The mapping of
book id to title and author (in English) is in the book_ids dictionary below.

This script will then do the following:

1. Filter out questions that are not in English.
2. Filter out questions that are not about who the murderer/culprit is. (show what fraction was kept)
3. Filter out individual questions that occur before 85% position in the story
4. Filter out novels where ALL remaining questions occur before 85% position (to focus on late-reveal mysteries)
5. Write out the data to datasets/detectiveqa/, 
   Note I want one .json file per book, and one .json file for the collection.
   IMPORTANT: Note there are 172 samples in the dataset, but only 62 novels. So we will have 62 json files, and the
   "questions" list will aggregate from multiple samples.

   Specifically we will have:
   - a collection.json file with keys "domain", "source", "created", "num_items", "total_documents", "description", "items" (list of book ids, which are just str(ints)).
   - in items/, write each book as a json file named <book_id>.json, with keys: (TODO: I think we should also include the novel_id, novel_title, novel_author, novel_text, time_cost, num_paragraphs)
     - "title-and-author": str (obtained from book_ids)
     - "text": str (obtained from data-source/detectiveqa_novels/<book_id>.txt, verbatim)
     - "questions": list of dicts with keys:
       - "question": str (obtained from the questions list in the sample)
       - "answer": str (obtained from the questions list in the sample)
       - "reasoning": List[str] (obtained from the questions list in the sample)
       - "answer_position": int (obtained from the questions list in the sample)
       - "clue_position": List[int] (obtained from the questions list in the sample)
       - "options": dict (mapping A, B, C, D to str) (obtained from the questions list in the sample)
       - "distraction": dict (mapping A, B, C, D to str) (obtained from the questions list in the sample)

Note structure of the ORIGINAL data we are loading from is:
# 172 shards in the data
dataset = load_dataset("Phospheneser/DetectiveQA", streaming=True)
samples = list(dataset.take(172)) #TODO is this correct or am I missing any data?
# Structure: each sample is a dict with keys:
# - novel_id: int
# - num_paragraphs: int (Let's not use this for now)
# - time_cost
# - questions: list of dicts with keys:
#   - question: str
#   - answer: str
#   - reasoning: List[str]
#   - answer_position: int
#   - clue_position: List[int]
#   - options: dict (mapping A, B, C, D to str)
#   - distraction: dict (mapping A, B, C, D to str); TODO not sure what this is for.
"""

import json
import os
from pathlib import Path
from collections import defaultdict
from datetime import datetime
#from datasets import load_dataset
from langdetect import detect
from difflib import SequenceMatcher
import re

# Note: my attempt to get these in English may not be completely accurate
# https://huggingface.co/datasets/Phospheneser/DetectiveQA/tree/main/novel_data_en
book_ids = dict([
    (100, "Alphabet Puzzle - Seiichiro Oyama"),
    (103, "The Murder on the Links - Agatha Christie"),
    (104, "The Kidnapped Prime Minister - Agatha Christie"),
    (105, "The Mystery of the Blue Train - Agatha Christie"),
    (106, "Peril at End House - Agatha Christie"),
    (107, "Death in the Clouds - Agatha Christie"),
    (108, "Three Act Tragedy - Agatha Christie"),
    (109, "Cards on the Table - Agatha Christie"),
    (110, "Murder in Mesopotamia - Agatha Christie"),
    (114, "Hercule Poirot's Christmas - Agatha Christie"),
    (116, "Sad Cypress - Agatha Christie"),
    (117, "Evil Under the Sun - Agatha Christie"),
    (118, "Taken at the Flood - Agatha Christie"),
    (120, "After the Funeral - Agatha Christie"),
    (121, "Hickory Dickory Dock - Agatha Christie"),
    (124, "The Clocks - Agatha Christie"),
    (126, "Halloween Party - Agatha Christie"),
    (127, "Elephants Can Remember - Agatha Christie"),
    (128, "Curtain: Poirot's Last Case - Agatha Christie"),
    (130, "The Murder at the Vicarage - Agatha Christie"),
    (132, "The Body in the Library - Agatha Christie"),
    (133, "By the Pricking of My Thumbs - Agatha Christie"),
    (134, "A Murder is Announced - Agatha Christie"),
    (136, "4.50 from Paddington - Agatha Christie"),
    (137, "The Mirror Crack'd from Side to Side - Agatha Christie"),
    (138, "A Caribbean Mystery - Agatha Christie"),
    (140, "Sleeping Murder - Agatha Christie"),
    (142, "The Secret of Chimneys - Agatha Christie"),
    (144, "The Seven Dials Mystery - Agatha Christie"),
    (145, "The Witness for the Prosecution - Agatha Christie"),
    (149, "Sparkling Cyanide - Agatha Christie"),
    (15, "The Spider Man - Edogawa Ranpo"),
    (150, "Crooked House - Agatha Christie"),
    (151, "Three Blind Mice - Agatha Christie"),
    (16, "The Magician - Edogawa Ranpo"),
    (198, "The ABC Murders - Agatha Christie"),
    (203, "One, Two, Buckle My Shoe - Agatha Christie"),
    (209, "Mrs McGinty's Dead - Agatha Christie"),
    (219, "The Monogram Murders - Sophie Hannah"),
    (241, "Why Didn't They Ask Evans? - Agatha Christie"),
    (25, "The Roman Hat Mystery - Ellery Queen"),
    (252, "Murder by Command - Agatha Christie"),
    (26, "The Greek Coffin Mystery - Ellery Queen"),
    (27, "The Egyptian Cross Mystery - Ellery Queen"),
    (28, "The Tragedy of X - Ellery Queen"),
    (29, "The Tragedy of Y - Ellery Queen"),
    (30, "The Tragedy of Z - Ellery Queen"),
    (31, "The Three Coffins - John Dickson Carr"),
    (33, "The Crooked Hinge - John Dickson Carr"),
    (40, "The Klein Bottle - Okajima Futari"),
    (53, "The Dutch Shoe Mystery - Ellery Queen"),
    (56, "Drury Lane's Last Case - Ellery Queen"),
    (79, "Winged Darkness - Maya Yutaka"),
    (81, "The One-Eyed Girl - Yutaka Maya"),
    (82, "The Disease of Murder - Takemaru Abiko"),
    (83, "The Two-Headed Devil - Arisu Arisugawa"),
    (84, "The Decagon House Murders - Yukito Ayatsuji"),
    (87, "Everything Becomes F - Hiroshi Mori"),
    (90, "The Hyperbolic Murder - Kyotaro Nishimura"),
    (93, "The Red Museum - Seiichiro Oyama"),
    (97, "The Scottish Game - Yasuhiko Nishizawa"),
    (99, "The Night She Died - Yasuhiko Nishizawa")
])


# Question mappings to make questions more natural
QUESTION_MAPPINGS = {
    " Who killed Ximian Li ( )": "Who killed Ximian Li?",
    " Which of the following individuals is most likely to be the instigator of the entire case?": "Who was the main culprit?",
    " The murderer of the Fosse brothers must be ( ).": "Who murdered the Fosse brothers?",
    " The real culprit behind the scene in Guanxue Pavilion is ( )": "Who was the real culprit behind the scene in Guanxue Pavilion?",
    " The murderer of Olga Semenova is ( )": "Who murdered Olga Semenova?",
    " The real killer in this serial murder case ( )": "Who was the real killer in this serial murder case?",
    " Who is the true killer of Helen?": "Who killed Helen?",
    " The real killer of Alyona is ( ) ": "Who killed Alyona?",
    " The killer of Yuecai is": "Who killed Yuecai?",
}

def is_english(text):
    """Check if text is in English using langdetect."""
    try:
        return detect(text) == 'en'
    except:
        # If detection fails, assume it's not English
        return False


def is_murderer_question(question):
    """Check if question is about who the murderer/culprit is."""
    question_lower = question.lower()
    
    # Allowlist: Questions that should ALWAYS be considered murder questions
    # even if they don't contain murder-related keywords
    allowlist_phrases = [
        # Add specific questions or phrases here that should be included
        "Who kidnapped Donohue?",
        "Who caused the death of Rosalind Claude?",
        "Who is the real mastermind behind these murder cases in the text?",
        "Who planned the death of Mrs. Simington?",
        "Who was Ronnie DeFrancis killed by?",
        "Who was Ruth killed by?",
        "Which of the following individuals is most likely to be the instigator of the entire case?",
        "Who pushed Castles off the cliff?",
        "Who is the mastermind behind the murder of Mrs. Franklin?",
        "The person who directly killed Harington Pace was",
        "Who was Anne Morris killed by?",
        "Who poisoned Polly?",
        "Who is involved in the murder of Colonel Prothero?",
        #"Who poisoned Marina's coffee?",
    ]
    
    # Check if question matches any allowlist phrase (force include)
    for allowed_phrase in allowlist_phrases:
        if allowed_phrase.lower() in question_lower:
            return True
    
    # Blocklist: Questions that should NOT be considered murder questions
    # even if they contain murder-related keywords
    blocklist_phrases = [
        # Add specific questions or  phrases here that should be excluded
        "Who killed George?", # same killer as other q
        "The killer of the plan to kill Kaneko is", #confusing question, what did they mean
        "Who was the murderer of Inoke?",
        "Who was the murderer of Yadon?",
        "Who could be the murderer of Chatham?",
        "Who was the murderer of the Klaudok family?",
        "Who is the real killer of Aristide?", #same killer as other q
        "Who is the murderer of my grandfather?", #confusingly written and repeat
        "Who is the suspect of killing Giuseppe?",
        "Who poisoned Marina's coffee?",
        "Who killed Mrs. Stanley?",
        "Who is the killer of Celia?",
        "Who killed Agnes, the maid of the Chingmingtong family?",
        "Who killed Pierre Frey?",
        "Who was the main perpetrator of the murder of Mr. Corley?",
        "Who killed Wood?", # suspect is wrong
        "Who was the murderer in the Joyce murder case?", # partial repeat killer and not sure if totally correct
        " Murderer of Lily ( )", # same killer as other
        "Who was the person who made the toad-faced man commit suicide?", #same killer as other q
        "The murderer of Mrs. Marshall is ( ).", # same murderer as Alyona, so ok to remove
        "There are commonalities among the three cases, and the inferred perpetrator is ( ).", #wrong culprit name
        "Miss Marple believes who the real culprit is (",
        "Which of the following individuals is most likely to be the instigator of the entire case?",
        "Who has the greatest suspicion of being the murderer",# not the actual killer, but a suspect
        "According to Agnes's statement, who killed Mr. Muller?", #don't want "according to..." questions
        "Who is the mastermind behind the multiple murders that took place in the mansion according to Bog?", #don't want "according to..." questions
        "According to Calgary's reasoning, who do you think killed Philip?", #don't want "from perspective of..." questions
        "Who manipulates others through psychological suggestion to commit crimes in the text?", #not essential
        "How did the killer kill Richard Abernathy",
        "Based on the current clues, who is the killer (",
        "Who is the murderer of Mary?", # early suspect, not the actual murderer
        "How did the killer leave the scene?",
        "Mr. Marshall believes that who killed Mrs. Aggles?", # avoid "from perspective of..." questions
        "Who killed Mrs. Upward?", #Duplicate
        "What is the true identity of the killer?", #Duplicate
        "From the perspective of the police inspector, who killed Mrs. Ascher?", # happens too early in the novel, and we don't want "from perspective of.. " questions
        "The murderer of the deceased is (", #Duplicate
        "Who was the murderer of Alfred and Harold?", # Duplicate
        "Who killed Joyce?", #Duplicate
        "Who is the mastermind behind the murder of Mrs. Franklin?", #Near Duplicate
        "Who was the killer of Celia Austin?", # 'Who was the killer of Celia' is already there
        "Based on the current clues, who is the killer (", #duplicate, checked only eliminates one
        "Based on the existing clues, who is the perpetrator of the serial murder case (", #Duplicate, checked only eliminates one
        "Who is the perpetrator of the serial murder case?", #Duplicate, checked only eliminates one
        "The real culprit of the triple murder case 18 years ago was who?", #Malformed answer options, remove
        "The murderer of Lily is", #Duplicate
        "The killer of Helen is (", #Duplicate
        "Who was the killer of Mrs. Boyle?", #Duplicate
        "Who is the suspect of killing Heather?",
        "Who killed Heather?", #Duplicate
        "What is the purpose of the killer sending an invitation letter to the doctoral film screening?",
        "What method did the killer plan to use to create an alibi after the crime?",
        "Why was Yamamura Mariko injured by the murderer?",
        " Who is the person who planned to use themselves as bait to lure the killer into revealing themselves?",
        "Among the following professions, the one that matches the killer is",
        "Who could not be the murderer?",
        "Davenport suspects Gregory Dyson's attitude is",
        "What was the motive behind the murderer's killing of Colonel Partridge?",
        "Which of the following individuals is a suspect in the murder of Victoria?",
        "Based on the current clues, the following person is not likely to be the killer",
        "Who could not be the killer of Simon Lee",
        "When investigating the theft of the diamond, who was the first suspect",
        "Dr. Roberts believes the killer is",
        "The murderer chose which game of bridge to commit the murder of Mr. Chaturvedi",
        "Why is there no suspect in a murder scene but no one is suspected?",
        "Where is the murderer hiding?",
        'Who is the "one o\'clock" that Bond suspects?',
        "Who does Detective Batter suspect of attempting to steal the files?",
        "What did the perpetrator do after the incident?",
        "Who is responsible for Moira's disappearance?",
        "Who is not the killer of Senator Fudge?",
        "Which hand did the killer use to stab the senator with the knife?",
        "The number of suspects in this murder case is",
        "Apart from the real culprit, another person involved in the novel is",
        "In connection with the death of Inna Brennt, which of the following individuals is a suspect?",
        "Based on the analysis of the crime scene, what is the inferred method of poisoning used by the perpetrator?",
        "Who is the suspect in the murder of Rosemary?",
        "In the anonymous letter of Little Chai brothers, what kind of criminal technique was mentioned?",
        "How was it determined that the killer had a close connection with the Dutch Memorial Hospital?",
        "Who is the nationality of the murderer?",
        "Who could not be the killer of Mrs. Ladden?",
        "Who is the suspect in the murder of Richard Lochhead?",
        "Who was the person who called Mr. Poirot for help?",
        "What was the motive of the criminal?",
        "How did the murderer take Field's hat?",
        "The motive of the killer leaving poisoned pears",
        "Who was the actual witness of the murder case",
        "Based on the current clues, who would definitely not be the killer of Joyce",
        "Who was the person who actually witnessed the murder case",
        "Based on the current clues, who could possibly not be the murderer",
        "Who could not be the perpetrator of the murder that occurred at the Brougham Hotel",
        "Who was the person who called Adwick Fenn with the intention of extortion",
        "Who is not the killer in this serial murder case",
        "When Mukae arrived at the Inari family, who was the chairman",
        "Why did the killer kill Celia Austin?",
        "What measures did the killer take to conceal their actions of murdering Celia Austin?",
        "What is the purpose of the killer sending an invitation letter to the doctoral film screening?",
        "What is the key factor that enables the killer to commit the crime so smoothly?",
        "The killer's criteria for selecting victims during the serial killings were:",
        "The principle of selecting victims by ABC murderer should be:",
        "How did the killer ABC choose the victims?",
        "The reason why the killer mistakenly identified the target person was",
        "What is the profession of the murderer?",
        "After a murder case occurred in the cinema, who did the police suspect to be the killer?",
        "Why didn't the killer arrange the victims in alphabetical order in the movie theater?",
        "Why isn't Castor the killer in the Baxter case?",
        "Miss Gilchrist is suspected of being involved in which incident in the novel?",
        "Who couldn't possibly be the murderer when Horiuchi was killed?",
        "Which of the following individuals is not likely to be the murderer of Mrs. Upward?",
        "The person who pushed Polo off the railway platform, the impossible suspect is?",
        "Where did the killer hide the body?",
        "Who was the true identity of the deceased?",
        "Police suspect the deceased to be linked to",
        "Why did the killer choose to use a lighter for illumination?",
        "The hideout of the criminal is",
        "Hoyt wonders who killed Rachel?",
        "Mr. Marshall thinks the killer is",
        "Who does the police suspect killed Philip?",
        "The initial conclusion states, who killed Mrs. Aggles?",
    ]
    
    # Check if question matches any blocklist phrase
    for blocked_phrase in blocklist_phrases:
        if blocked_phrase.lower() in question_lower:
            return False
    
    # Check for murder/culprit keywords
    murderer_keywords = [
        'murderer', 'killer', 'culprit', 'perpetrator', 'criminal',
        'who killed', 'who murdered', 'who committed', 'who did it',
        'who is responsible', 'who was the', 'guilty person', 'suspect'
    ]
    return any(keyword in question_lower for keyword in murderer_keywords)


def similarity(a, b):
    """Calculate similarity between two strings."""
    return SequenceMatcher(None, a, b).ratio()


def clean_option_text(text):
    """Clean corrupted option text by removing translation artifacts and long quotes."""
    if not isinstance(text, str):
        return text
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # If it contains "Translation:" pattern, extract just the name before the colon
    if ': "' in text and "(Translation: \"I'm here to listen" in text:
        # Extract just the name before the colon
        name_part = text.split(':')[0].strip()
        return name_part
    
#    # If it's very long (>50 chars) and contains quotes, likely corrupted
    if len(text) > 50 and '"' in text:
        # Try to extract just the first word/name
        words = text.split()
        if words:
            # Take first word, remove any leading punctuation
            first_word = words[0].lstrip(' "\'')
            return first_word
    
    return text



def main():
    """Main ingestion process."""
    print("🔍 Starting DetectiveQA ingestion...")
    
    # Create output directories
    output_dir = Path("datasets/detectiveqa")
    items_dir = output_dir / "items"
    output_dir.mkdir(exist_ok=True)
    items_dir.mkdir(exist_ok=True)
    
    # Open log files for inspection
    murder_questions_file = open("murder_questions.txt", "w", encoding="utf-8")
    non_murder_questions_file = open("non_murder_questions.txt", "w", encoding="utf-8")
    warnings_file = open("ingestion_warnings.txt", "w", encoding="utf-8")
    
    # Load annotation files directly (bypassing streaming dataset corruption)
    print("📥 Loading annotation files...")
    annotation_dir = Path("data-source/detectiveqa-annotations")

    aisup_dir = annotation_dir / "AIsup_anno"
    human_dir = annotation_dir / "human_anno"
    
    if not annotation_dir.exists():
        print("❌ Annotation files not found!")
        print("💡 Run: python download_annotation_files.py first")
        murder_questions_file.close()
        non_murder_questions_file.close()
        warnings_file.close()
        return
    
    # Load all annotation files
    samples = []
    
    # Load AIsup_anno files
    aisup_files = list(aisup_dir.glob("*.json"))
    print(f"📂 Found {len(aisup_files)} AIsup_anno files")
    
    for file_path in aisup_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                file_data = json.load(f)
                # The files contain arrays of objects, so extend the samples list
                if isinstance(file_data, list):
                    samples.extend(file_data)
                else:
                    samples.append(file_data)
        except Exception as e:
            warnings_file.write(f"ERROR: Could not load annotation file {file_path}: {e}\n")
    
    # Load human_anno files
    human_files = list(human_dir.glob("*.json"))
    print(f"📂 Found {len(human_files)} human_anno files")
    
    for file_path in human_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                file_data = json.load(f)
                # The files contain arrays of objects, so extend the samples list
                if isinstance(file_data, list):
                    samples.extend(file_data)
                else:
                    samples.append(file_data)
        except Exception as e:
            warnings_file.write(f"ERROR: Could not load annotation file {file_path}: {e}\n")
    
    print(f"✅ Loaded {len(samples)} samples from annotation files")
    
    # Group samples by novel_id
    print("📚 Grouping samples by novel...")
    novels_data = defaultdict(list)
    for sample in samples:
        novels_data[sample['novel_id']].append(sample)
    
    print(f"📊 Found {len(novels_data)} unique novels")
    
    # Process each novel
    total_questions = 0
    english_questions = 0
    murder_questions = 0
    processed_novels = []
    filtered_by_position = []  # Track novels filtered out by 85% threshold
    questions_filtered_by_position = 0  # Track individual questions filtered by 85% threshold
    
    for novel_id, novel_samples in novels_data.items():
        print(f"\n📖 Processing novel {novel_id}...")
        
        # Check if we have the book mapping
        if novel_id not in book_ids:
            warnings_file.write(f"WARNING: Novel ID {novel_id} not found in book_ids mapping\n")
            continue
        
        # Load novel text
        novel_file = Path(f"data-source/detectiveqa/{novel_id}-*.txt")
        novel_files = list(Path("data-source/detectiveqa").glob(f"{novel_id}-*.txt"))
        
        if not novel_files:
            warnings_file.write(f"WARNING: Novel text file not found for ID {novel_id}\n")
            continue
        
        novel_text_file = novel_files[0]
        try:
            with open(novel_text_file, 'r', encoding='utf-8') as f:
                novel_text = f.read()
        except Exception as e:
            warnings_file.write(f"ERROR: Could not read novel file {novel_text_file}: {e}\n")
            continue
        
        # Aggregate questions from all samples for this novel
        all_questions = []
        seen_questions = []  # For duplicate detection
        
        for sample in novel_samples:
            for question_data in sample['questions']:
                total_questions += 1
                
                # Check if English
                if not is_english(question_data['question']):
                    continue
                english_questions += 1
                
                # Check for duplicates/near-duplicates
                question_text = question_data['question']
                for seen_q in seen_questions:
                    if similarity(question_text, seen_q) > 0.8:  # 80% similarity threshold
                        warnings_file.write(f"WARNING: Potential duplicate question in novel {novel_id}:\n")
                        warnings_file.write(f"  Existing: {seen_q}\n")
                        warnings_file.write(f"  New: {question_text}\n\n")
                        break
                seen_questions.append(question_text)
                
                # Check if it's about murderer/culprit
                if is_murderer_question(question_data['question']):
                    murder_questions += 1
                    murder_questions_file.write(f"Novel {novel_id}: {question_data['question']}\n")
                    
                    # Clean options to remove corruption
                    cleaned_options = {
                        key: clean_option_text(value) 
                        for key, value in question_data['options'].items()
                    }
                    
                    # Check for specific duplicate question that needs to be filtered
                    question_text = question_data['question'].strip()
                    should_skip = False
                    
                    if question_text == "The real killer of Alyona is ( )":
                        # Remove the duplicate with specific distraction option
                        distraction_a = question_data['distraction'].get('A', '')
                        if "Elina's husband, knowing Elina's affair, has a motive to kill" in distraction_a:
                            print(f"  🗑️  Removing duplicate question in novel {novel_id}: {question_text}")
                            print(f"     (filtering based on distraction option A)")
                            should_skip = True
                    
                    if not should_skip:
                        # Apply question mapping if exists
                        original_question = question_data['question']
                        mapped_question = QUESTION_MAPPINGS.get(original_question, original_question)
                        
                        # Log when a mapping is applied
                        if mapped_question != original_question:
                            print(f"  📝 Mapping question in novel {novel_id}:")
                            print(f"     From: {original_question}")
                            print(f"     To:   {mapped_question}")
                        
                        # Add to filtered questions
                        all_questions.append({
                            "question": mapped_question,
                            "answer": question_data['answer'],
                            "reasoning": question_data['reasoning'],
                            "answer_position": question_data['answer_position'],
                            "clue_position": question_data['clue_position'],
                            "options": cleaned_options,
                            "distraction": question_data['distraction']
                        })
                else:
                    non_murder_questions_file.write(f"Novel {novel_id}: {question_data['question']}\n")
        
        # Skip novels with no qualifying questions
        if not all_questions:
            print(f"  ⏭️  Skipping novel {novel_id} (no qualifying questions)")
            continue
        
        # Filter out individual questions that occur before 85% position threshold
        lines = novel_text.split('\n')
        total_lines = len([line for line in lines if line.strip().startswith('[') and ']' in line])
        
        if total_lines > 0:
            original_count = len(all_questions)
            filtered_questions = []
            
            for question in all_questions:
                answer_position = question.get('answer_position', 0)
                normalized_position = answer_position / total_lines
                
                if normalized_position >= 0.85:  # Keep questions at or after 85%
                    filtered_questions.append(question)
            
            all_questions = filtered_questions
            questions_filtered = original_count - len(all_questions)
            questions_filtered_by_position += questions_filtered
            
            if questions_filtered > 0:
                print(f"  🔍 Novel {novel_id}: Filtered out {questions_filtered} early questions (before 85%)")
        
        # Check if all questions were filtered out by position threshold
        if not all_questions:
            novel_title = book_ids[novel_id]
            print(f"  🚫 Filtering out novel {novel_id} ({novel_title})")
            print(f"     All questions occur before 85% position")
            filtered_by_position.append({
                'novel_id': novel_id,
                'title': novel_title,
                'num_questions': 0  # All questions were filtered
            })
            continue
        
        print(f"  ✅ Novel {novel_id}: {len(all_questions)} qualifying questions")
        
        # Create item JSON following README structure
        item_data = {
            "item_metadata": {
                "item_id": str(novel_id),
                "num_documents": 1
            },
            "documents": [
                {
                    "doc_id": f"{novel_id}",
                    "content": novel_text,
                    "metadata": {
                        "title": book_ids[novel_id].split(" - ")[0],
                        "author": book_ids[novel_id].split(" - ")[1],
                        "novel_id": novel_id,
                        "num_paragraphs": novel_samples[0]['num_paragraphs'],
                        "time_cost": novel_samples[0]['time_cost'],
                        "questions": all_questions
                    }
                }
            ]
        }
        
        # Write item file
        item_file = items_dir / f"{novel_id}.json"
        with open(item_file, 'w', encoding='utf-8') as f:
            json.dump(item_data, f, indent=2, ensure_ascii=False)
        
        processed_novels.append(str(novel_id))
    
    # Create collection.json
    collection_data = {
        "domain": "detective novel",
        "source": "DetectiveQA dataset - Phospheneser/DetectiveQA from Hugging Face",
        "created": datetime.now().isoformat(),
        "num_items": len(processed_novels),
        "total_documents": len(processed_novels),
        "description": "Detective novels with questions about murderer/culprit identification, filtered from DetectiveQA dataset. Excludes individual questions that occur before 85% position and novels where all questions occur before 85% position, focusing on late-reveal mysteries.",
        "items": processed_novels
    }
    
    collection_file = output_dir / "collection.json"
    with open(collection_file, 'w', encoding='utf-8') as f:
        json.dump(collection_data, f, indent=2, ensure_ascii=False)
    
    # Close log files
    murder_questions_file.close()
    non_murder_questions_file.close()
    warnings_file.close()
    
    # Print summary
    print(f"\n📊 INGESTION SUMMARY:")
    print(f"Total questions processed: {total_questions}")
    print(f"English questions: {english_questions} ({english_questions/total_questions*100:.1f}%)")
    print(f"Murder/culprit questions: {murder_questions} ({murder_questions/english_questions*100:.1f}% of English)")
    print(f"Individual questions filtered by 85% position: {questions_filtered_by_position}")
    print(f"Novels processed: {len(processed_novels)}")
    
    # Report on filtered novels
    if filtered_by_position:
        print(f"\n🚫 NOVELS FILTERED BY 85% POSITION THRESHOLD:")
        print(f"   {len(filtered_by_position)} novels filtered out (all questions before 85% position)")
        for filtered_novel in filtered_by_position:
            print(f"   • Novel {filtered_novel['novel_id']}: {filtered_novel['title']}")
            print(f"     (all questions were early)")
    else:
        print(f"\n✅ No novels completely filtered by 85% position threshold")
    
    print(f"\nOutput directory: {output_dir}")
    print(f"\n📝 Log files created:")
    print(f"  - murder_questions.txt: {murder_questions} questions about murderers/culprits")
    print(f"  - non_murder_questions.txt: {english_questions - murder_questions} other English questions")
    print(f"  - ingestion_warnings.txt: Warnings and errors during processing")


if __name__ == "__main__":
    main()
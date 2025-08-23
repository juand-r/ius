#!/usr/bin/env python3
"""
DetectiveQA Content Name Corrections

This module provides comprehensive name correction for story content based on
the question/answer mappings and manual analysis. It fixes character names
in the actual story text to match the corrected names in questions.
"""

import re
import json
from typing import Dict, List, Tuple, Set
from pathlib import Path

# Comprehensive name corrections based on question/answer mappings and analysis
CONTENT_NAME_CORRECTIONS = {
    # From Novel 219 - The Monogram Murders by Sophie Hannah
    "Richard Nixon": "Richard Negus",
    "Nixon": "Negus",
    
    # From Novel 149 - Sparkling Cyanide by Agatha Christie  
    "Rosamary Barron": "Rosemary Barton",
    "Rosamary": "Rosemary",
    "Barron": "Barton",
    
    # From Novel 144 - The Seven Dials Mystery by Agatha Christie
    "Ronnie DeFrancis": "Ronnie Devereux", 
    "DeFrancis": "Devereux",
    "De Vere": "Devereux",
    
    # From Novel 137 - The Mirror Crack'd from Side to Side by Agatha Christie
    "Heather Barducci": "Heather Badcock",
    "Barducci": "Badcock",
    "Bardcock": "Badcock",
    "Bardock": "Badcock", 
    "Badecock": "Badcock",
    "Badecock's": "Badcock's",
    "Badekirk": "Badcock",
    "Bardcock": "Badcock",
    "Bardwell": "Badcock",
    
    # From Novel 124 - The Clocks by Agatha Christie
    "Quentin Ducassin": "Quentin Duguesclin",
    "Ducassin": "Duguesclin",
    "Dugastlin": "Duguesclin",
    "Durgeslin": "Duguesclin",
    
    # From Novel 126 - Halloween Party by Agatha Christie
    "Olga Semenova": "Olga Seminoff",
    "Semenova": "Seminoff",
    
    # From Novel 130 - The Murder at the Vicarage by Agatha Christie
    "Colonel Proctor": "Colonel Protheroe",
    "Proctor": "Protheroe",
    
    # From answer corrections (excluded novels, but including for completeness)
    "Jachi Hate": "York Hatter",
    "Jachi": "York",  # Only in specific context
    "Hamnet Sedlak": "Hamnet Sadler", 
    "Sedlak": "Sadler",
    
    # Additional answer corrections from ANSWER_MAPPINGS
    "Nayton": "Nayton",
    "Johns Hopkins": "Nurse Hopkins",
    "Jesse Hopinks": "Jessie Hopkins",
    "Sheriff Trott": "Sheriff Trotter",
    "Trott": "Trotter", 
    "Trotterer": "Trotter",
    "Troolt": "Trotter",
    "Troote": "Trotter",
    "Throttle": "Trotter",
    "Mrs. Aggleton": "Mrs. Aggles",
    "Aggleton": "Aggles",
    "Agle": "Aggles",
    
    # Kristen variants
    "Corlston": "Kristen",
    "Corton": "Kristen",
    "Corston": "Kristen",
    
    # From other corrections discovered
    "Rosalind Claude": "Rosaleen Cloade",
    "Rosalind": "Rosaleen",
    "Claude": "Cloade",
    "Claudet": "Cloade",
    "Clude": "Cloade", 
    "Claud": "Cloade",
    "Mary Gerald": "Mary Gerrard",
    "Gerald": "Gerrard", 
    "Mrs. Laidner": "Mrs. Leidner",
    "Laidner": "Leidner",
    "Anne Morris": "Anne Meredith",
    "Morris": "Meredith",
    "Alan Carstairs": "Alan Carstairs",  # This one might be correct
    "Castles": "Carstairs",
    
    # Additional manual corrections from user
    "Zenjirouo": "Zenjiro",
    "Jang Shengzhang": "Jiang Shengzhang",
    "Jian Shenxiang": "Jiang Shenxiang",
    "Vaughan": "Vaughn",
    "Gwendolyn Vance": "Gwendolyn Vaughn",
    "Barnet-French": "Bassington-French",
    "Baskington-French": "Bassington-French", 
    "Basington-French": "Bassington-French",
    "Bashington-French": "Bassington-French",
    "Prichard": "Pritchard",
    "Ada Gransberry": "Ada Grantsberry",
    "Ada Gransbury": "Ada Grantsberry",
    "Ada Gransberry's": "Ada Grantsberry's",
    "Margaret Enst": "Margaret Ernst", 
    "Hendrson": "Henderson",
    "Bonaparte Caste": "Bonaparte Castel",
    "Bonaparte Caster": "Bonaparte Castel",
    "Catherine Royal": "Catherine Royale",
    "Franklin Clark": "Franklin Clarke",
    "Doveton": "Doverton",
    "Christopher Rain": "Christopher Raine",
    "Christopher Rine": "Christopher Raine",
    "Aristide": "Aristides",
    "Harvillan": "Harvilland",
    "Havilland": "Harvilland",
    "Patrician": "Patricia",
    "Sassage": "Sassiage",
    "Beetman": "Betman",
    "Bettman": "Betman",
    "Detective Batter": "Detective Batters",
    "Superintendent Batter": "Superintendent Battle",
    "Jackie Averycliffe": "Jackie Averycliff",
    "Edith Pargette": "Edith Pargett", 
    "Lily Abbott": "Lily Abbot",
    "Miss Maple": "Miss Marple",
    "Jane Maple": "Jane Marple",
    "Walter Fin": "Walter Fain",
    "Estervile": "Esterville",
    "Luckesmoor": "Lucksmoor",
    "Lethlean": "Lethlan",
    "Emott": "Emmott",
    "Emot": "Emott",
    "Monnier": "Monier",
    "Welman": "Wellman",
    "Rodrick": "Roderick",
    "Rodric": "Roderick", 
    "Redfearn": "Redfern",
    "Corigan": "Corrigan",
    "Thomlinson": "Tomlinson",
    "Hilary": "Hillary",
    "Hilar": "Hillary",
    "Weber": "Webber",
    "Pemerton": "Pemmerton", 
    "Fulleton": "Fullerton",
    "Llewellyn-Smith": "Llewelyn-Smith",
    "Mackay": "McKay",
    "Reidley": "Ridley",
    "Lestranges": "Lestrange",
    "Cartlington": "Carlington",
    "Cartington": "Carlington",
    "Cardington": "Carlington",
    "Craighorne": "Craignorn",
    "Chinchcliffe": "Chinchcliff",
    "Ashebrooke": "Ashebrook",
    "Richfield": "Ritchfield",
    "Firrell": "Firrel",
    "Betson": "Bettson",
    "Preyscott": "Prescott",
    "Jonathan Parry": "Jonathan Parr",
    "Gosington": "Gossington",
    "Bruster": "Bruste",
}

# Specific novel mappings for context-sensitive replacements
NOVEL_SPECIFIC_CORRECTIONS = {
    "103": {  # Murder on the Links (Poirot mystery with many OCR errors)
        # Character name corrections from o3 analysis
        "Polo": "Poirot",
        "Poro": "Poirot", 
        "Jack Reno": "Jack Renault",
        "Jack Reynold": "Jack Renault", 
        "Jack Reynolds": "Jack Renault",
        "Mrs Reno": "Mrs Renault",
        "Madame Reno": "Madame Renault",
        "Mme Renouf": "Mme Renault",
        "Mrs Rennie": "Mrs Renault",
        "Renouf": "Renault",
        "Reno": "Renault",
        "Reynold": "Renault",
        "Reynolds": "Renault",
        "Gabriel Stoner": "Gabriel Stonor",
        "Stoner": "Stonor", 
        "Guillot": "Giraud",
        "Gilead": "Giraud",
        "Gillette": "Giraud",
        "Guillaud": "Giraud", 
        "Giles": "Giraud",
        "Gerrard": "Giraud",
        "Gabor": "Giraud",
        "Gilroy": "Giraud",
        "Galloway": "Giraud",
        "Joliot": "Giraud",
        "Bexes": "Bex",
        "Bexs": "Bex",
        "Bax": "Bex",
        "Baxter": "Bex",
        "Alt": "Hautet",
        "Alte": "Hautet", 
        "Alté": "Hautet",
        "Altair": "Hautet",
        "Altman": "Hautet",
        "Arter": "Hautet",
        "Artès": "Hautet",
        "Arte": "Hautet",
        "Ardouin": "Hautet",
        "Artier": "Hautet",
        "Artères": "Hautet",
        "Dubroil": "Daubreuil",
        "Dobroil": "Daubreuil",
        "Dobroël": "Daubreuil",
        "Dubrol": "Daubreuil",
        "Dubor": "Daubreuil",
        "Duboroff": "Daubreuil",
        "Dobrova": "Daubreuil",
        "Doborl": "Daubreuil",
        "Dobrel": "Daubreuil",
        "Doboril": "Daubreuil", 
        "Dobréol": "Daubreuil",
        "Dobrull": "Daubreuil",
        "Dobroslav": "Daubreuil",
        "Dobrow": "Daubreuil",
        "Dobrowolsky": "Daubreuil",
        "Dobrowolski": "Daubreuil",
        "Dobroy": "Daubreuil",
        "Dubois": "Daubreuil",
        "Dobré's": "Daubreuil's",
        "Dobré": "Daubreuil",
        "Marta": "Marthe",
        "Marte": "Marthe",
        "Mart": "Marthe",
        "Malt": "Marthe", 
        "Merta": "Marthe",
        "Milda": "Marthe",
        "DuVeen": "Duveen",
        "DuVine": "Duveen",
        "DuVain": "Duveen",
        "DuVen": "Duveen",
        "DuVigne": "Duveen", 
        "Devenish": "Duveen",
        "Cono": "Conneau",
        "Conno": "Conneau",
        "Connon": "Conneau",
        "Conolly": "Conneau",
        "Konopalski": "Conneau",
        "Cornet": "Conneau",
        "Mallow": "Malart",
        "Melvilleville": "Merlinville",
        "Melvinville": "Merlinville",
        "Gennevieve": "Genéviève",
    },
         "83": {  # Japanese mystery novel with transcription errors
        # Character name corrections from o3 analysis
        "Uma Ryuji": "Okuma Ryūji",
        "Ma Rong-san": "Ma Long-san",
        "Masaya": "Matsuri",  # Note: Keep Masaya for male student separate
        "Jang Shin": "Jingū-ji",
        "Jang God": "Jingū-ji",
        "Jang Godo": "Jingū-ji",
        "Jiang Shen": "Jingū-ji",
        "Jiang Godo": "Jingū-ji",
     },
    "219": {  # The Monogram Murders
        "Nixon": "Negus",
        "Richard Nixon": "Richard Negus",
        "Naygreaves": "Negus",
        "Naylegh": "Negus", 
        "Naylors": "Negus",
        "Neagles": "Negus",
        "Nehy": "Negus",
        "Neville": "Negus",
        "Nichol": "Negus",
        "Nielsen": "Negus",
        "Nigel": "Negus",
        "Nigel's": "Negus's",
        "Nigg": "Negus",
        "Nigg's": "Negus's", 
        "Nigger": "Negus",
        "Nigges": "Negus",
        "Nigges'": "Negus's",
        "Nigges's": "Negus's",
        "Niggles": "Negus",
        "Nigus": "Negus",
        "Nigus's": "Negus's",
        "Nix": "Negus",
        "Nix's": "Negus's",
        "Nixes": "Negus",
        "Nixey": "Negus",
        "Janie Hobbs": "Jennie Hobbs",
        "Janie": "Jennie",
        "Jane": "Jennie",
        "Jenny": "Jennie",
        "Janice": "Jennie",
    },
    "149": {  # Sparkling Cyanide
        "Rosamary": "Rosemary", 
        "Barron": "Barton",
        "Rosamary Barron": "Rosemary Barton",
    },
    "144": {  # The Seven Dials Mystery
        "DeFrancis": "Devereux",
        "Ronnie DeFrancis": "Ronnie Devereux",
        "De Vere": "Devereux",
        "De Vere's": "Devereux's",
        "DeFleur": "Devereux",
        "DeVere": "Devereux",
        "DeVille": "Devereux", 
        "DeVille's": "Devereux's",
    },
    "137": {  # The Mirror Crack'd from Side to Side
        "Barducci": "Badcock",
        "Heather Barducci": "Heather Badcock",
        "Gregg": "Gregorovna",
        "Gregor": "Gregorovna",
        "Gregorovitch": "Gregorovna",
    },
    "124": {  # The Clocks
        "Ducassin": "Duguesclin",
        "Quentin Ducassin": "Quentin Duguesclin",
    },
    "126": {  # Halloween Party
        "Semenova": "Seminoff",
        "Olga Semenova": "Olga Seminoff",
        "Olgar": "Olga",
    },
    "130": {  # The Murder at the Vicarage
        "Proctor": "Protheroe", 
        "Colonel Proctor": "Colonel Protheroe",
        "Anne Prothero": "Anne Protheroe",
        "Anne Prothero's": "Anne Protheroe's",
        "Leith Prothero": "Leith Protheroe", 
        "Letitia Prothero": "Letitia Protheroe",
        "Miss Prothero": "Miss Protheroe",
    },

    "118": {  # Approved corrections from o3 analysis
        "Croud": "Cloade",
        "Rolly": "Ronnie",
        "Croude": "Cloade",
        "Crouse": "Cloade",
        "Jeremy Cloude": "Jeremy Cloade",
        "Jeremy Clowde": "Jeremy Cloade",
        "Jeremy Croude": "Jeremy Cloade",
        "Leonel": "Lionel",
        "Katharine": "Katherine",
        "Kathryn": "Katherine",
        "Rosaline": "Rosaleen",
        "Rosalyn": "Rosaleen",
        "Rozalin": "Rosaleen",
        "Rozaline": "Rosaleen",
        "Rosalin": "Rosaleen",
        "Andreh": "AndrÃ©",
        "Andrah": "AndrÃ©",
        "Andrau": "AndrÃ©",
        "Andeau": "AndrÃ©",
        "Andeh": "AndrÃ©",
        "Andrich": "AndrÃ©",
        "Andeah": "AndrÃ©",
        "Andhare": "AndrÃ©",
        "Ander": "AndrÃ©",
        "Potter": "Porter",
        "Pretor": "Porter",
        "Enock": "Enoch",
        "Inok": "Enoch",
        "Yates": "Yate",
        "Yaton": "Yate",
        "Yad on": "Yate",
        "Yatton": "Yate",
        "Spencer": "Spence",
        "Spenser": "Spence",
        "Greves": "Graves",
        "Greiv": "Graves",
        "Grevs": "Graves",
        "Grieve": "Graves",
        "Pollock": "Poirot",
        "Poro": "Poirot",
        "Johnny Vavasu": "Johnny Vavaser",
        "Vavasseur": "Vavaser",
        "Trentron": "Trentham",
        "Froxbank": "Frognack",
        "Frobank": "Frognack",
        "Frockbank": "Frognack",
        "Frobenk": "Frognack",
        "Froebel": "Frognack",
    },
    "133": {  # Approved corrections from o3 analysis
        "Jerry Bardon": "Jerry Burton",
        "Jerry Bates": "Jerry Burton",
        "Jerry Bateson": "Jerry Burton",
        "Joan": "Joanna",
        "Joanne": "Joanna",
        "Joan-na": "Joanna",
        "Bardon": "Barton",
        "Barden": "Barton",
        "Borton": "Barton",
        "Batten": "Barton",
        "Button": "Barton",
        "Bates": "Barton",
        "Griffiths": "Griffith",
        "Greaves": "Griffith",
        "Griffith": "Griffith",
        "Griffitts": "Griffith",
        "Simmington": "Symmington",
        "Smington": "Symmington",
        "Shimmington": "Symmington",
        "Shingeton": "Symmington",
        "Chinnington": "Symmington",
        "Chimmington": "Symmington",
        "Schmington": "Symmington",
        "Smythington": "Symmington",
        "Smythson": "Symmington",
        "Meghan": "Megan",
        "Pei": "Pye",
        "Pay": "Pye",
        "Poy": "Pye",
        "Poynter": "Pye",
        "Little Fitz": "Little Firs",
        "Little Fitze": "Little Firs",
        "Little Fritz": "Little Firs",
    },
    "26": {  # Approved corrections from o3 analysis
        "Kagisu": "Kagarice",
        "Kagis": "Kagarice",
        "Kagi": "Kagarice",
        "Kagist": "Kagarice",
        "Kagistu": "Kagarice",
        "Kakistos": "Kagarice",
        "Kaxistos": "Kagarice",
        "Kaskisu": "Kaskist",
        "Keswick": "Kaskist",
        "Green Shaw": "Greenshaw",
        "Slone": "Sloane",
        "Slon": "Sloane",
        "Slonim": "Sloane",
        "Shlonan": "Shloan",
        "Shlon": "Shloan",
        "Sholan": "Sholon",
        "Slocum": "Sholon",
        "Mrs. Sloane": "Mrs. Sloan",
        "Mrs. Slocum": "Mrs. Sloan",
        "Blithe": "Blyth",
        "Blight": "Blyth",
        "Blayt": "Blyth",
        "Woz": "Wotz",
        "Wozz": "Wotz",
        "Wozniak": "Wotz",
        "Woss": "Wotz",
        "Vos": "Wotz",
        "Voss": "Wotz",
        "Stewarts": "Stuart",
        "Eada": "Ada",
        "Aida": "Ada",
        "Henrywell": "Henneville",
        "Henriwell": "Henneville",
    },
    "90": {  # Approved corrections from o3 analysis
        "Kiyoko": "Kyoko",
        "Gouda": "Gōda",
        "Teshima": "Tajima",
    },
    "110": {  # Approved corrections from o3 analysis
        "Lethern": "Leatheran",
        "Lethlan": "Leatheran",
        "Lethan": "Leatheran",
        "Lesterland": "Leatheran",
        "Latham": "Leatheran",
        "Reilly": "Riley",
        "Rely": "Riley",
        "Liley": "Riley",
        "Raeley": "Riley",
        "Raley": "Riley",
        "Shira": "Shirley",
        "Sheila": "Shirley",
        "Sherrie": "Shirley",
        "Liddner": "Leidner",
        "Liddon": "Leidner",
        "Leden": "Leidner",
        "Laddner": "Leidner",
        "Ladner": "Leidner",
        "Ladden": "Leidner",
        "Ledner": "Leidner",
        "Laidner": "Leidner",
        "Ledna": "Leidner",
        "Ledene": "Leidner",
        "Lavini": "Lavigny",
        "Lavigne": "Lavigny",
        "Lavyne": "Lavigny",
        "Lavine": "Lavigny",
        "Larrine": "Lavigny",
        "Carry": "Carey",
        "Karry": "Carey",
        "Kerry": "Carey",
        "Kari": "Carey",
        "Emott": "Emmott",
        "Lettel": "Littell",
        "Mocado": "Moccado",
        "Macado": "Moccado",
        "McAdoo": "Moccado",
        "Mukaddo": "Moccado",
        "Mocadlo": "Moccado",
        "Mocador": "Moccado",
        "Mocadu": "Moccado",
        "Meltrum": "Melton",
        "Penryman": "Penniman",
        "Kelsy": "Kelsey",
        "Kelsh": "Kelsey",
        "Hasani": "Hasanie",
        "Hachani": "Hasanie",
        "Hazzani": "Hasanie",
        "Hussani": "Hasanie",
    },
    "33": {  # Approved corrections from o3 analysis
        "Baro": "Barrow",
        "Barroa": "Barrow",
        "Farlay": "Farley",
        "Fairly": "Farley",
        "Lili": "Lily",
        "Gowl": "Gaul",
        "Gole": "Gaul",
        "Goule": "Gaul",
        "Murrey": "Murray",
        "Murry": "Murray",
        "Muri": "Murray",
        "Madeleine": "Madeline",
        "Dary": "Dare",
        "Darely": "Dare",
        "Hapworth": "Hapgood",
        "Hapwood": "Hapgood",
        "Harpwood": "Hapwood",
        "Weik": "Vake",
        "Weikai": "Vake",
        "Wei-Kai": "Weikai",
        "Wick": "Vake",
        "Weck": "Vake",
    },
    "29": {  # Approved corrections from o3 analysis
        "Quincy": "Quincey",
        "Abbey": "Abigail",
        "Abuco": "Abigail",
        "Abucock": "Abigail",
        "Abouco": "Abigail",
        "Abdul": "Abigail",
        "Aboukir": "Abigail",
        "Smita": "Smith",
        "Smithe": "Smith",
    },
    "209": {  # Approved corrections from o3 analysis
        "Brodini": "Brodny",
        "Broadwee": "Brodny",
        "Brodini-": "Brodny",
    },
    "100": {  # Approved corrections from o3 analysis
        "Funahara": "Fukuhara",
        "Fuyuama": "Fukuhara",
        "Fuji-hara": "Fukuhara",
        "Rie Hui": "Riehui",
        "Sengaku": "Senkaku",
        "Sensaku": "Senkaku",
        "Calgarth": "Calgary",
    },
    "252": {  # Approved corrections from o3 analysis
        "Calgar": "Calgary",
        "Argyle": "Aggles",
        "Aigle": "Aggles",
        "Agher": "Aggles",
        "Aggert": "Aggles",
        "Aggul": "Aggles",
        "Aghel": "Aggles",
        "Agar": "Aggles",
        "Gwendalyn": "Gwendolyn",
        "Gwendraith": "Gwendolyn",
        "Gwendra": "Gwendolyn",
        "Wain": "Vaughn",
        "Vane": "Vaughn",
        "Lindstrom": "Lindström",
        "Lindstrum": "Lindström",
        "Coulston": "Kirsten",
        "Corton": "Kirsten",
        "Colston": "Kirsten",
        "Caulston": "Kirsten",
        "Coleson": "Kirsten",
        "Custon": "Kirsten",
        "Darland": "Durrant",
        "Darlant": "Durrant",
        "Darwent": "Durrant",
        "Darrent": "Durrant",
        "Darrant": "Durrant",
        "Dallant": "Durrant",
        "McMaster": "MacMaster",
    },
}

# All corrections are now defined manually above

class ContentNameCorrector:
    """Handles name corrections in story content."""
    
    def __init__(self):
        self.replacements_made = []
        self.validation_snippets = []
    
    def correct_names_in_text(self, text: str, novel_id: str = None) -> str:
        """
        Apply name corrections to text content.
        
        Args:
            text: The text content to correct
            novel_id: Optional novel ID for context-specific corrections
            
        Returns:
            Corrected text with character names fixed
        """
        corrected_text = text
        
        # Apply novel-specific corrections first (most precise)
        if novel_id and novel_id in NOVEL_SPECIFIC_CORRECTIONS:
            for incorrect, correct in NOVEL_SPECIFIC_CORRECTIONS[novel_id].items():
                if incorrect in corrected_text:
                    # Find a context snippet for validation
                    self._add_validation_snippet(corrected_text, incorrect, correct, novel_id)
                    corrected_text = self._replace_name_carefully(corrected_text, incorrect, correct)
                    self.replacements_made.append({
                        "novel_id": novel_id,
                        "type": "novel_specific",
                        "from": incorrect,
                        "to": correct,
                        "occurrences": text.count(incorrect)
                    })
        
        # Apply general corrections (broader)
        for incorrect, correct in CONTENT_NAME_CORRECTIONS.items():
            if incorrect in corrected_text:
                # Skip if we already handled this in novel-specific corrections
                if novel_id and novel_id in NOVEL_SPECIFIC_CORRECTIONS:
                    if incorrect in NOVEL_SPECIFIC_CORRECTIONS[novel_id]:
                        continue
                
                self._add_validation_snippet(corrected_text, incorrect, correct, novel_id)
                corrected_text = self._replace_name_carefully(corrected_text, incorrect, correct)
                self.replacements_made.append({
                    "novel_id": novel_id or "unknown",
                    "type": "general",
                    "from": incorrect,
                    "to": correct,
                    "occurrences": text.count(incorrect)
                })
        
        return corrected_text
    
    def _replace_name_carefully(self, text: str, incorrect: str, correct: str) -> str:
        """
        Replace names carefully, respecting word boundaries where appropriate.
        """
        # For full names (containing spaces), do exact replacement
        if " " in incorrect:
            return text.replace(incorrect, correct)
        
        # For single names, use word boundary replacement to avoid partial matches
        # But be careful with punctuation and context
        pattern = r'\b' + re.escape(incorrect) + r'\b'
        return re.sub(pattern, correct, text)
    
    def _add_validation_snippet(self, text: str, incorrect: str, correct: str, novel_id: str):
        """Add a snippet showing the replacement for validation."""
        pos = text.find(incorrect)
        if pos != -1:
            start = max(0, pos - 50)
            end = min(len(text), pos + len(incorrect) + 50) 
            before_snippet = text[start:end]
            after_snippet = before_snippet.replace(incorrect, correct)
            
            self.validation_snippets.append({
                "novel_id": novel_id or "unknown",
                "from": incorrect,
                "to": correct,
                "before": before_snippet,
                "after": after_snippet,
                "position": pos
            })
    
    def get_replacement_summary(self) -> Dict:
        """Get summary of all replacements made."""
        return {
            "total_replacements": len(self.replacements_made),
            "replacements": self.replacements_made,
            "validation_snippets": self.validation_snippets[:10]  # First 10 for review
        }
    
    def save_validation_report(self, filepath: str):
        """Save validation report to JSON file."""
        report = {
            "summary": self.get_replacement_summary(),
            "all_validation_snippets": self.validation_snippets
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    
    def reset(self):
        """Reset tracking for new processing session."""
        self.replacements_made = []
        self.validation_snippets = []


def test_corrections():
    """Test the correction system on some sample text."""
    corrector = ContentNameCorrector()
    
    test_cases = [
        ("219", "Richard Nixon was found dead. Nixon had been shot."),
        ("149", "Rosamary Barron was poisoned. Mrs. Barron died quickly."),
        ("144", "Ronnie DeFrancis was murdered. DeFrancis was young."),
        ("137", "Heather Barducci was killed. Barducci was the victim."),
    ]
    
    print("Testing Content Name Corrections:")
    print("=" * 50)
    print(f"Total manual corrections: {len(CONTENT_NAME_CORRECTIONS)}")
    
    for novel_id, test_text in test_cases:
        corrected = corrector.correct_names_in_text(test_text, novel_id)
        print(f"\nNovel {novel_id}:")
        print(f"Original: {test_text}")
        print(f"Corrected: {corrected}")
    
    print(f"\nReplacements made: {len(corrector.replacements_made)}")
    for replacement in corrector.replacements_made:
        print(f"  {replacement['from']} → {replacement['to']} ({replacement['occurrences']} times)")


if __name__ == "__main__":
    test_corrections()
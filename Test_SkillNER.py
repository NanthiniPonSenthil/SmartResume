import spacy
from spacy.matcher import PhraseMatcher
from skillNer.skill_extractor_class import SkillExtractor

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Create PhraseMatcher (do NOT use 'attr' if your SpaCy version doesn't support it)
phrase_matcher = PhraseMatcher(nlp.vocab)  # just pass the vocab

# Define your custom skills
skills_db = ["Python", "Java", "AWS", "Docker", "Kubernetes", "SQL", "machine learning"]

# Initialize SkillExtractor
extractor = SkillExtractor(nlp, skills_db=skills_db, phraseMatcher=phrase_matcher)

# Example text
text = """
John has experience in Python, Java, SQL, AWS, and Docker. 
He has also worked with Kubernetes and machine learning.
"""

# Extract skills
skills = extractor.extract_skills(text)

print("Extracted skills:", skills)

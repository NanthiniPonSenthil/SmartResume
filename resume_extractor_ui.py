import re
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime
import os
import google.generativeai as genai

import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox, ttk
from spacy.matcher import PhraseMatcher
from skillNer.skill_extractor_class import SkillExtractor
from skillNer.general_params import SKILL_DB


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import spacy
try:
    NLP = spacy.load('en_core_web_sm')
    # Initialize PhraseMatcher with vocab
    phrase_matcher = PhraseMatcher(NLP.vocab)

except OSError:
    print("Downloading 'en_core_web_sm' model...")
    from spacy.cli import download
    download('en_core_web_sm')
    NLP = spacy.load('en_core_web_sm')

# =========================
# Resume Data Model
# =========================
@dataclass
class ResumeData:
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    skills: List[str] = None
    experience: List[str] = None
    education: List[str] = None
    overall_experience_years: Optional[float] = None
    certifications: List[str] = None
    languages: List[str] = None
    confidence_score: float = 0.0

    def __post_init__(self):
        # Initialize empty lists if None
        for field in ['skills', 'experience', 'education', 'certifications', 'languages']:
            if getattr(self, field) is None:
                setattr(self, field, [])

    def to_dict(self) -> Dict:
        return asdict(self)

    def calculate_confidence(self) -> float:
        """Calculate confidence score based on extracted data completeness"""
        score = 0.0
        weights = {
            'name': 0.15,
            'email': 0.15,
            'phone': 0.10,
            'skills': 0.20,
            'experience': 0.25,
            'education': 0.10,
            # 'technologies' removed
        }
        
        if self.name: score += weights['name']
        if self.email: score += weights['email']
        if self.phone: score += weights['phone']
        if self.skills: score += weights['skills'] * min(1.0, len(self.skills) / 5)
        if self.experience: score += weights['experience'] * min(1.0, len(self.experience) / 3)
        if self.education: score += weights['education'] * min(1.0, len(self.education) / 2)
    # technologies removed from model
        
        self.confidence_score = round(score * 100, 1)
        return self.confidence_score


# =========================
# Enhanced Resume Extractor
# =========================
class EnhancedResumeExtractor:
    def __init__(self):
        self.nlp = NLP
        self.phrase_matcher = phrase_matcher
        self.tech_keywords = self._load_tech_keywords() 
        self.skill_keywords = self._load_skill_keywords()
        self.certification_keywords = self._load_certification_keywords()
        self.language_keywords = self._load_language_keywords()
        # Optional SkillNER integration: try multiple import names and keep a reference.
        import importlib
        self.skillner = None
        for candidate in ('skillner', 'skillNer', 'SkillNer'):
            try:
                mod = importlib.import_module(candidate)
                # Support common shapes: class SkillnerExtractor, module-level APIs, or callable
                if hasattr(mod, 'SkillnerExtractor'):
                    try:
                        self.skillner = mod.SkillnerExtractor()
                    except Exception:
                        self.skillner = mod
                else:
                    self.skillner = mod
                logger.info(f"SkillNER loaded from module: {candidate}")
                break
            except ModuleNotFoundError:
                # Try next candidate name
                continue
            except Exception as e:
                logger.warning(f"SkillNER import attempt for {candidate} failed: {e}")
                continue

    def _load_skill_keywords(self) -> List[str]:
        return [
            "python", "java", "c++", "c#", "javascript", "sql", "excel", "power bi", 
            "aws", "azure", "docker", "kubernetes", "react", "angular", "node.js", 
            "machine learning", "data analysis", "project management", "leadership", 
            "communication", "problem solving", "teamwork", "agile", "scrum",
            "database design", "system architecture", "devops", "ci/cd"
        ]

    
    def _load_tech_keywords(self) -> Dict[str, str]:
        skills = [
        "python", "java", "c++", "c#", "javascript", "typescript", "react", "angular",
        "vue", "node.js", "django", "flask", "spring", "dotnet", ".net", ".net core",
        "asp.net", "asp.net core", "entity framework", "web api", "ajax", "sql", "mysql",
        "postgresql", "mongodb", "oracle", "aws", "azure", "gcp", "docker", "kubernetes",
        "git", "jenkins", "linux", "windows", "html", "css", "sass", "power bi", "tableau",
        "spark", "hadoop", "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch",
        "rest api", "graphql", "selenium", "jira", "confluence", "matlab", "sas", "r",
        "php", "swift", "go", "rust", "scala", "bash", "shell", "redis", "elasticsearch",
        "firebase", "android", "ios", "xcode", "visual studio", "unity", "unreal", "salesforce",
        "sap", "abap", "powerapps", "servicenow", "bigquery", "looker", "airflow", "terraform",
        "excel", "machine learning", "data analysis", "project management", "leadership",
        "communication", "problem solving", "teamwork", "agile", "scrum", "database design",
        "system architecture", "devops", "ci/cd", "eclipse", "shell scripting", "apache tomcat",
        "nagios", "splunk", "ansible", "kotlin", "dart", "perl", "julia", "rpa", "uipath",
        "next.js", "nuxt.js", "svelte", "express.js", "laravel", "ruby on rails", "cassandra",
        "neo4j", "mariadb", "firebase realtime database", "snowflake", "redshift", "bigtable",
        "helm", "argocd", "istio", "prometheus", "grafana", "vault", "nomad", "databricks",
        "keras", "lightgbm", "xgboost", "hive", "apache kafka", "apache nifi", "cypress",
        "postman", "appium", "robot framework", "jmeter", "trello", "asana", "monday.com",
        "webpack", "babel", "docker compose", "vmware", "hyper-v", "powershell"
    ]


            # Convert to dict so SkillExtractor can use it
        return {skill: "TECH" for skill in set(skills)}


    def _load_certification_keywords(self) -> List[str]:
        return [
           'aws certified', 'azure certified', 'google cloud certified', 'cissp', 'cism',
    'pmp', 'scrum master', 'scrum master certified (scrum alliance)', 'six sigma', 'itil',
    'comptia', 'ccna', 'ccnp', 'ccie', 'mcse',
    'microsoft certified', 'oracle certified', 'salesforce certified', 'certified kubernetes', 'docker certified',
    'red hat certified', 'aws certified solutions architect', 'aws certified developer', 'aws certified sysops administrator', 'aws certified devops engineer',
    'aws cloud practitioner', 'azure fundamentals', 'azure administrator associate', 'azure solutions architect expert', 'azure security engineer associate',
    'azure devops engineer expert', 'google cloud associate cloud engineer', 'google cloud professional cloud architect', 'oscp', 'comptia security+',
    'comptia network+', 'prince2', 'lean six sigma', 'pmi-acp', 'safe agilist',
    'vmware certified professional', 'microsoft certified: data analyst associate', 'google data analytics professional certificate', 'snowflake snowpro', 'red hat certified engineer',
    'salesforce certified administrator', 'salesforce certified developer'
        ]  
    


    def _load_language_keywords(self) -> List[str]:
        return [
            'english', 'spanish', 'french', 'german', 'italian', 'portuguese', 
            'chinese', 'japanese', 'korean', 'russian', 'arabic', 'hindi', 
            'dutch', 'swedish', 'norwegian', 'danish'
        ]

    def extract(self, text: str) -> ResumeData:
        """Main extraction method with improved error handling"""
        data = ResumeData()
        # Process text with spaCy
        doc = NLP(text)
        # Extract all fields
        data.name = self._extract_name(doc)
        data.email = self._extract_email(text)
        data.phone = self._extract_phone(text)
        data.skills = self._extract_skills(text)
        data.experience = self._extract_experience(text)
        data.education = self._extract_education(text)
        # technologies field removed; skills are extracted by _extract_skills
        data.certifications = self._extract_certifications(text)
        data.languages = self._extract_languages(text)
        data.overall_experience_years = self._extract_overall_experience(text)
        return data

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for better extraction"""
        # Remove excessive whitespace and normalize line breaks
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n', text)
        return text.strip()

    def _extract_name(self, doc) -> Optional[str]:
        """Extract name using spaCy's NER."""
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                return ent.text
        return None

    def _extract_email(self, text: str) -> Optional[str]:
        """Extract email using regex."""
        pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
        match = re.search(pattern, text)
        return match.group(0) if match else None

    def _extract_phone(self, text: str) -> Optional[str]:
        """Extract phone number using regex."""
        pattern = re.compile(r"(\+?\d{1,2}[-.\s]?)?(\(?\d{3}\)?[-.\s]?)?(\d{3}[-.\s]?\d{4})")
        match = re.search(pattern, text)
        return match.group(0) if match else None

    def _extract_skills(self, text: str) -> List[str]:
        """Extract skills using SkillNER if available; otherwise fallback to keyword/regex.

        The method expects SkillNER to expose one of these shapes:
        - module-level function: extract_skills(text)
        - instance method: extract(text) or predict(text)
        - callable object returning an iterable of skill strings

        Returns a deduplicated, title-cased list limited to 15 items.
        """
        # Use SkillNER if available — collect its output but don't return immediately.
        skillner_skills = []
        if getattr(self, 'skillner', None) is not None:
            try:
                skills_out = []
                doc = NLP(text)
                exceptions = [".NET", ".net", ".Net"]
                skills_out = [
                        token.text for token in doc
                        if token.text in exceptions
                        or (
                            not token.is_stop
                            and not token.like_num
                            and token.pos_ != "PRON"
                            and token.ent_type_ not in [ "DATE"]
                            and any(char.isalpha() for char in token.text)
                        )
                    ]

                # Clean and title-case
                normalized = []
                for s in skills_out:
                    try:
                        s_str = str(s).strip()
                        if s_str:
                            normalized.append(s_str.title())
                    except Exception:
                        continue

                # Dedupe preserving order
                seen = set()
                unique = []
                for s in normalized:
                    k = s.lower()
                    if k not in seen:
                        seen.add(k)
                        unique.append(s)

                skillner_skills = unique
            except Exception as e:
                logger.warning(f"SkillNER extraction failed ({e}); falling back to keyword extraction")

        # # Fallback logic: original keyword + regex extraction
        # found_skills = []
        # text_lower = text.lower()

        # for skill in self.skill_keywords:
        #     if re.search(rf'\b{re.escape(skill.lower())}\b', text_lower):
        #         found_skills.append(skill.title())

        # # Merge SkillNER results with keyword-based found skills (keyword results are reliable for clear tokens)
        # merged = []
        # seen_final = set()
        # for s in (skillner_skills + found_skills):
        #     key = s.lower()
        #     if key not in seen_final:
        #         seen_final.add(key)
        #         merged.append(s)

        # # Look for skills section
        # skills_match = re.search(r'(skills?|competencies|expertise)([\s\S]{0,300})', text, re.IGNORECASE)
        # if skills_match:
        #     skills_section = skills_match.group(2)
        #     additional_skills = re.findall(r'[•\-\*]?\s*([A-Za-z][A-Za-z\s&/\-]{2,20})', skills_section)
        #     found_skills.extend([skill.strip().title() for skill in additional_skills if skill.strip()])

        # # Remove duplicates while preserving order for the found_skills (from keyword/section extraction)
        # seen = set()
        # unique_skills = []
        # for skill in found_skills:
        #     if skill and skill.lower() not in seen:
        #         seen.add(skill.lower())
        #         unique_skills.append(skill)

        # Combine SkillNER results with the cleaned unique keyword/section results
        final = []
        seen_final = set()
        for s in (skillner_skills ):
            if not s:
                continue
            key = s.lower()
            if key not in seen_final:
                seen_final.add(key)
                # normalize title-casing for display
                final.append(s.title())

        return final

    

    def _extract_certifications(self, text: str) -> List[str]:
        """Extract professional certifications"""
        found_certs = []
        
        for cert in self.certification_keywords:
            if re.search(rf'\b{re.escape(cert)}\b', text, re.IGNORECASE):
                found_certs.append(cert.title())
        
        # Look for certification patterns
        cert_patterns = [
            r'(certified?\s+[A-Za-z\s]+(?:professional|specialist|expert|associate))',
            r'([A-Z]{2,6}\s+certified)',
        ]
        
        for pattern in cert_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]
                found_certs.append(match.strip().title())

        # Preserve order and remove duplicates
        return list(dict.fromkeys(found_certs))

    def extract_skills_and_certs(self, text: str) -> Tuple[List[str], List[str]]:
        """Return (skills, certifications) for reuse (no duplicates of logic)."""
        skills = self._extract_skills(text) if text else []
        certs = self._extract_certifications(text) if text else []
        return skills, certs

    def extract_jd_sections(self, text: str) -> Tuple[List[str], List[str]]:
        """Extract 'Must-Have' and 'Good To Have' sections from a JD text.

        Returns (must_have_list, good_to_have_list). Items are normalized strings.
        Handles headings like 'Must-Have', 'Must Have', 'Good To Have', 'Nice to have',
        and captures bullets, comma-separated lists, and inline sentences.
        """
        must = []
        good = []
        if not text:
            return must, good

        # Normalize line endings and split
        lines = [l.rstrip() for l in text.splitlines()]

        current = None
        # common heading starts
        must_heads = re.compile(r'^(must[\s\-]*have[s]?|must[:\-]?$|must\s*\:)', re.IGNORECASE)
        good_heads = re.compile(r'^(good[\s\-]*to[\s\-]*have|nice\s+to\s+have|preferred|desired|good[:\-]?)', re.IGNORECASE)
        # headings that should terminate a must/good section when encountered.
        # Only match explicit section headers (e.g. 'Experience' or 'Experience:') so
        # sentences that start with the word 'Experience' (like 'Experience working in...')
        # are NOT treated as new section starters.
        other_section_starts = re.compile(
            r'^(?:experience\s*:?$|responsibilities\b|responsibilities:\b|qualifications\b|education\b|benefits\b|about\s+the\s+role\b|role\s+responsibilities\b|required\s+experience\b|required\s+experience\s*\(years\)|required:)',
            re.IGNORECASE,
        )

        # We'll collect raw lines for must/good sections and reuse _extract_skills
        def collect_section_text(line: str) -> Optional[str]:
            # remove leading bullet markers and excessive punctuation
            l = re.sub(r'^[\-•\*\s]+', '', line).strip()
            if not l:
                return None

            # Remove common leading boilerplate that describes expectations
            l = re.sub(r'^(?:candidate should have knowledge of|candidate should have|candidate should have experience in|hands on exp in|hands on experience in|experience in working in|experience working in|experience working on|experience in|experience with|knowledge of|must have|should have|must be able to|must have experience in)\s+', '', l, flags=re.IGNORECASE)

            # Remove trailing noise phrases like 'also expected' or stray words
            l = re.sub(r'\b(?:also expected(?: to be)?|also required|also expected to have|expected(?: to be)?|and expected)\b\.?', '', l, flags=re.IGNORECASE).strip()

            # remove leftover trailing punctuation
            l = l.strip().strip(':;.')

            # If the remaining text is very short and looks like a heading token, skip
            if re.match(r'^(must|good|preferred|desired|nice to have)$', l.strip().lower()):
                return None

            return l

        for ln in lines:
            s = ln.strip()
            if not s:
                # ignore blank lines but keep current section active so
                # content following an empty line is still captured under the heading
                continue

            low = s.lower()
            # If a heading line
            if must_heads.search(low):
                # capture any inline items after ':' on same line as raw text
                after = re.split(r':', s, maxsplit=1)
                if len(after) > 1 and after[1].strip():
                    t = collect_section_text(after[1])
                    if t:
                        must.append(t)
                current = 'must'
                continue
            if good_heads.search(low):
                after = re.split(r':', s, maxsplit=1)
                if len(after) > 1 and after[1].strip():
                    t = collect_section_text(after[1])
                    if t:
                        good.append(t)
                current = 'good'
                continue

            # If the current line looks like another major section, stop capturing must/good
            if other_section_starts.search(s):
                current = None
                continue

            # If we are within a section, accumulate raw lines
            if current == 'must':
                t = collect_section_text(s)
                if t:
                    must.append(t)
                continue
            if current == 'good':
                t = collect_section_text(s)
                if t:
                    good.append(t)
                continue

        # Fallback: if no explicit headings, try to detect common phrasing lines and collect raw sentences
        if not must and not good:
            for sent in re.split(r'[\.\n;]+', text):
                s = sent.strip()
                if not s:
                    continue
                low = s.lower()
                if low.startswith('must') or low.startswith('should have') or 'must have' in low:
                    must.append(s)
                elif low.startswith('good to have') or low.startswith('nice to have') or low.startswith('preferred'):
                    good.append(s)

        # Use the existing _extract_skills logic on the concatenated section texts
        def normalize_section(raw_lines: List[str]) -> List[str]:
            if not raw_lines:
                return []
            joined = '\n'.join(raw_lines)
            # reuse the extractor's skill extraction (no hardcoding here)
            skills = self._extract_skills(joined)

            # Build allowed set from configured keyword lists to filter out junk tokens
            allowed = set(k.lower() for k in self.skill_keywords)
            # tech keywords is a dict mapping skill->category
            allowed.update(k.lower() for k in self.tech_keywords.keys())

            seen = set()
            out = []
            for s in skills:
                k = s.lower()
                # Keep skills that appear in our known lists (tech or skill keywords)
                if k in allowed:
                    if k not in seen:
                        seen.add(k)
                        out.append(s)
                else:
                    # Also allow some multi-word patterns or common extensions (e.g., '.net core', 'react')
                    # Keep items that contain letters and a minimum length and are not obviously junk
                    if re.search(r'[a-zA-Z]', s) and len(k) >= 3 and k not in {'candidate', 'knowledge', 'expected', 'hands', 'exp', 'working', 'good', 'tools'}:
                        if k not in seen:
                            seen.add(k)
                            out.append(s)

            return out

        return normalize_section(must), normalize_section(good)

    def _tokenize_skill(self, text: str) -> List[str]:
        """Normalize and tokenize a skill or phrase into meaningful tokens.

        Removes punctuation, common noise words, and returns lowercased tokens.
        """
        if not text:
            return []
        # lower, remove punctuation except plus signs (e.g., c++)
        s = text.lower()
        # replace common separators with spaces
        s = re.sub(r'[\./\\]+', ' ', s)
        # remove parentheses and colons
        s = re.sub(r'[\(\):,;]', ' ', s)
        # collapse non-alphanum to spaces (keep + for c++)
        s = re.sub(r'[^a-z0-9\+\s]', ' ', s)

        tokens = [t for t in re.split(r'\s+', s) if t]
        # noise words to ignore
        noise = {'also', 'expected', 'experience', 'experiencein', 'experiencein', 'working', 'methodology', 'hands', 'on', 'exp', 'candidate', 'should', 'have', 'knowledge', 'of', 'in', 'front', 'end', 'front-end', 'front_end', 'frontend', 'js'}
        # map some aliases
        alias_map = {'reactjs': 'react', 'react.js': 'react', 'netcore': 'net core', 'dotnetcore': 'net core', '.net': 'net', 'csharp': 'c#'}
        out = []
        for t in tokens:
            tt = t.strip()
            if not tt:
                continue
            if tt in noise:
                continue
            # apply alias mapping
            if tt in alias_map:
                mapped = alias_map[tt]
                for sub in mapped.split():
                    if sub and sub not in out:
                        out.append(sub)
                continue
            # keep tokens length >=2 or common ones like c++
            if len(tt) >= 2 or '++' in tt or '+' in tt:
                if tt not in out:
                    out.append(tt)
        return out

    def _jd_item_matches_resume(self, jd_item: str, resume_skills: List[str]) -> bool:
        """Return True if any token from jd_item appears in any resume skill tokens."""
        jd_tokens = set(self._tokenize_skill(jd_item))
        if not jd_tokens:
            return False
        # check against each resume skill token set
        for rs in resume_skills:
            rs_tokens = set(self._tokenize_skill(rs))
            if not rs_tokens:
                continue
            # if any jd token intersects resume tokens, treat as match
            if jd_tokens & rs_tokens:
                return True
            # also check if resume tokens include all jd tokens (strong match)
            if jd_tokens.issubset(rs_tokens):
                return True
        return False

    def _jd_cert_mandatory(self, text: str, certs: List[str]) -> str:
        """Determine if any certification mentioned in the JD is explicitly mandatory.

        Returns 'yes' if any certification appears in the same sentence as a mandatory
        keyword (must, required, mandatory, etc.), otherwise returns 'Optional'.
        """
        # Simpler deterministic logic: scan sentences for a certification token and a mandatory phrase.
        if not text:
            return 'Optional'

        mandatory_phrases = [
            'must be', 'must have', 'must', 'required', 'required to', 'mandatory',
            'is required', 'should be', 'must hold', 'must possess', 'required:'
        ]

        # Build a set of lowercase cert tokens from provided certs and from known keywords
        cert_tokens = set()
        if certs:
            for c in certs:
                if not c:
                    continue
                for tok in re.findall(r"[A-Za-z0-9]+", c.lower()):
                    if len(tok) >= 3:  # ignore very short tokens to reduce false positives
                        cert_tokens.add(tok)

        # include core tokens from certification_keywords to catch misspellings or alternate phrasing
        for cert_kw in self.certification_keywords:
            for tok in re.findall(r"[A-Za-z0-9]+", cert_kw.lower()):
                if len(tok) >= 3:
                    cert_tokens.add(tok)

        # Also include short canonical tokens for cloud providers and common certs
        extra = {'aws', 'azure', 'google', 'gcp', 'pmp', 'cissp', 'cism', 'docker', 'kubernetes'}
        cert_tokens.update(extra)

        # Break text into sentences and check
        sentences = re.split(r'[\.\n!?]+', text)
        for sent in sentences:
            s = sent.strip().lower()
            if not s:
                continue

            # quick check for 'certif' misspellings / mentions
            has_cert_mention = False
            if 'certif' in s or 'certificate' in s or 'certified' in s or 'cerification' in s:
                has_cert_mention = True

            # check for known cert tokens in the sentence using word boundaries
            for tok in cert_tokens:
                if re.search(rf"\b{re.escape(tok)}\b", s):
                    has_cert_mention = True
                    break

            if not has_cert_mention:
                continue

            # If any mandatory phrase is present in the same sentence, mark as required
            for phrase in mandatory_phrases:
                if phrase in s:
                    return 'yes'

        return 'Optional'

    def extract_jd_experience_requirement(self, text: str) -> Tuple[Optional[float], str]:
        """Extract a required years-of-experience value from the JD text and whether it's mandatory.

        Returns (years_required or None, 'yes'|'Optional').
        """
        if not text:
            return None, 'Optional'

        text_l = text.lower()

        # Try to capture an explicit 'Required Experience (years):' line first
        years = None
        m_exp_line = re.search(r'required\s+experience\s*\(years\)\s*:\s*([^\n\r]+)', text_l)
        exp_str = None
        if m_exp_line:
            exp_str = m_exp_line.group(1).strip()

        # If no explicit line, try to find common patterns nearby
        if not exp_str:
            # look for a short snippet after an 'experience' header
            m_snip = re.search(r'(experience[:\n\r].{0,120})', text_l)
            if m_snip:
                snip = m_snip.group(1)
                # accept ranges like '3-5', '3 – 5', or '3 to 5'
                m_range = re.search(r'(\d{1,2}(?:\.\d+)?)\s*(?:[-–—]|to)\s*(\d{1,2}(?:\.\d+)?)', snip, flags=re.IGNORECASE)
                if m_range:
                    exp_str = m_range.group(0)
                else:
                    m_num = re.search(r'(?:at least|min(?:imum)?|minimum|>=|>)?\s*(\d{1,2}(?:\.\d+)?)(?:\+)?\s*(?:years|yrs?)', snip)
                    if m_num:
                        exp_str = m_num.group(0)

        # Fallback: scan whole text for ranges or inequalities
        if not exp_str:
            # look for ranges like '8-13 years' or '8 to 13 yrs'
            m_range = re.search(r'(\d{1,2}(?:\.\d+)?)\s*(?:[-–—]|to)\s*(\d{1,2}(?:\.\d+)?)\s*(?:years|yrs?)', text_l, flags=re.IGNORECASE)
            if m_range:
                years = {'min': float(m_range.group(1)), 'max': float(m_range.group(2))}
            else:
                m_less = re.search(r'(?:less than|under|up to|<=)\s*(\d{1,2}(?:\.\d+)?)\s*(?:years|yrs?)', text_l)
                if m_less:
                    years = {'max': float(m_less.group(1))}
                else:
                    m_more = re.search(r'(?:at least|minimum|min(?:imum)?|>=|>)\s*(\d{1,2}(?:\.\d+)?)\s*(?:years|yrs?)', text_l)
                    if m_more:
                        years = {'min': float(m_more.group(1))}
                    else:
                        m_exact = re.search(r'(?<!\d)(\d{1,2}(?:\.\d+)?)\s*(?:years|yrs?)', text_l)
                        if m_exact:
                            years = {'exact': float(m_exact.group(1))}
                        else:
                            years = None
        else:
            s = exp_str
            # accept 'min to max' or 'min-max' formats in the explicit exp_str
            m_range = re.search(r'(\d{1,2}(?:\.\d+)?)\s*(?:[-–—]|to)\s*(\d{1,2}(?:\.\d+)?)', s, flags=re.IGNORECASE)
            if m_range:
                years = {'min': float(m_range.group(1)), 'max': float(m_range.group(2))}
            else:
                m_less = re.search(r'less than\s*(\d{1,2}(?:\.\d+)?)', s)
                if m_less:
                    years = {'max': float(m_less.group(1))}
                else:
                    m_more = re.search(r'(?:at least|minimum|min(?:imum)?|>=|>)\s*(\d{1,2}(?:\.\d+)?)', s)
                    if m_more:
                        years = {'min': float(m_more.group(1))}
                    else:
                        m_num = re.search(r'(\d{1,2}(?:\.\d+)?)', s)
                        if m_num:
                            years = {'exact': float(m_num.group(1))}
                        else:
                            years = None

        # Detect explicit 'Is Experience Mandatory: Yes' style flag
        is_mand = 'Optional'
        m_flag = re.search(r'is\s*experience\s*mandatory\s*:\s*(yes|y|true|required|mandatory|no|optional|false)', text_l)
        if m_flag:
            val = m_flag.group(1).strip()
            if val in ('yes', 'y', 'true', 'required', 'mandatory'):
                is_mand = 'yes'
            else:
                is_mand = 'Optional'
        else:
            # if words like 'must have' or 'required' appear near 'experience', treat as mandatory
            if re.search(r'(must have|required(?: to)?|minimum|min(?:imum)?|must)\s+\d', text_l):
                is_mand = 'yes'

        # If a closed range was detected and no explicit mandatory flag was present,
        # treat ranges like 3-5 or 4-5.5 as mandatory by default (user expectation).
        if is_mand == 'Optional' and isinstance(years, dict) and 'min' in years and 'max' in years:
            is_mand = 'yes'

        # Interpret 'must have N years' (without 'at least'/'minimum') as an exact requirement
        # so that resumes with > N years will fail when experience is mandatory.
        # If phrase contains 'at least' or 'minimum', keep as min.
        m_must = re.search(r"(?:(?:must have|must)\s*)(\d{1,2}(?:\.\d+)?)\s*(?:years|yrs?)", text_l)
        if m_must:
            # ensure not 'at least' style
            if not re.search(r'(at least|minimum|min(?:imum)?)\s*' + re.escape(m_must.group(1)), text_l):
                # set years to an exact value if not already a dict with min/max
                try:
                    val = float(m_must.group(1))
                    years = {'exact': val}
                except Exception:
                    pass

        return years, is_mand

    def _extract_languages(self, text: str) -> List[str]:
        """Extract spoken languages"""
        found_langs = []
        
        # Look for languages section
        lang_match = re.search(r'(languages?)([\s\S]{0,200})', text, re.IGNORECASE)
        if lang_match:
            lang_section = lang_match.group(2)
            for lang in self.language_keywords:
                if re.search(rf'\b{re.escape(lang)}\b', lang_section, re.IGNORECASE):
                    found_langs.append(lang.title())
        
        return list(set(found_langs))

    def _extract_overall_experience(self, text: str) -> Optional[float]:
        # Look for patterns like 'X years', 'X+ years', 'over X years', etc.
        matches = re.findall(r'(\d{1,2}(?:\.\d+)?)[+]?\s*(?:years?|yrs?)', text, re.IGNORECASE)
        years = [float(m) for m in matches if float(m) < 50]  # filter out unlikely values
        if years:
            return max(years)
        return None

    def _extract_experience(self, text: str) -> List[str]:
        """Enhanced work experience extraction"""
        results = []
        
        # Look for experience section
        exp_patterns = [
            r"((?:professional\s+)?experience|work\s+(?:history|experience)|employment(?:\s+history)?|career\s+(?:summary|history))([\s\S]+?)(?=\n(?:education|skills|projects|certifications)|$)",
        ]
        
        for pattern in exp_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                exp_section = match.group(2)
                
                # Enhanced job extraction patterns
                job_patterns = [
                    r'([A-Z][A-Za-z\s&/\-,\.]{5,50}?(?:Engineer|Developer|Manager|Analyst|Consultant|Lead|Director|Officer|Specialist|Architect|Designer|Scientist|Administrator|Coordinator|Executive|Intern|Associate|Senior))\s*(?:[-–—]|at|@)\s*([A-Z][A-Za-z0-9\s&.,\'\-]{3,40}?(?:Inc|LLC|Corp|Company|Ltd|Group|Systems|Technologies|Solutions)?)\s*(?:[-–—]|\(|\||\s)\s*(\d{4}(?:\s*[-–—]\s*(?:\d{4}|Present|Current))?)',
                    r'([A-Z][A-Za-z\s&/\-,\.]{5,50})\s*\n\s*([A-Z][A-Za-z0-9\s&.,\'\-]{3,40})\s*\n?\s*(\d{4}[\s\-–—]*(?:to|-)?\s*(?:\d{4}|Present|Current))',
                ]
                
                for job_pattern in job_patterns:
                    matches = re.findall(job_pattern, exp_section, re.IGNORECASE | re.MULTILINE)
                    for match in matches:
                        title, company, duration = match
                        # Clean up the extracted data
                        title = re.sub(r'[^\w\s&/\-]', '', title).strip()
                        company = re.sub(r'[^\w\s&.,\'\-]', '', company).strip()
                        duration = duration.strip()
                        
                        if len(title) > 5 and len(company) > 2:
                            results.append(f"{title} at {company} ({duration})")
        
        # Remove duplicates and filter
        seen = set()
        filtered_results = []
        for item in results:
            if item not in seen and 20 <= len(item) <= 150:
                seen.add(item)
                filtered_results.append(item)
        
        return filtered_results[:8]

    def _extract_education(self, text: str) -> List[str]:
        """Enhanced education extraction"""
        results = []
        
        # Enhanced education patterns
        patterns = [
            r'(Bachelor\s+of\s+[A-Za-z\s]+|B\.?(?:Tech|E|Sc|A|Com|S)\.?[A-Za-z\s]*|Master\s+of\s+[A-Za-z\s]+|M\.?(?:Tech|E|Sc|A|Com|S|BA)\.?[A-Za-z\s]*|Ph\.?D\.?|Doctor\s+of\s+[A-Za-z\s]+|MBA|MS|BS)\s+(?:in\s+)?([A-Za-z\s&,]+)?\s*(?:from\s+|at\s+|-\s+)?([A-Z][A-Za-z0-9\s&.,\'\-]+?(?:University|College|Institute|School|Academy))[^\n]*?(\d{4})',
            r'([A-Z][A-Za-z0-9\s&.,\'\-]+?(?:University|College|Institute|School|Academy))[^\n]*?(Bachelor|Master|B\.?Tech|M\.?Tech|Ph\.?D|MBA|B\.?Sc|M\.?Sc|BS|MS)[^\n]*?(\d{4})'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match) >= 3:
                    if len(match) == 4:
                        degree, field, institution, year = match
                        if field and field.strip():
                            results.append(f"{degree.strip()} in {field.strip()}, {institution.strip()}, {year}")
                        else:
                            results.append(f"{degree.strip()}, {institution.strip()}, {year}")
                    else:
                        institution, degree, year = match
                        results.append(f"{degree.strip()}, {institution.strip()}, {year}")
        
        # Remove duplicates
        return list(dict.fromkeys(results))[:5]


# =========================
# Enhanced File Reader
# =========================
class FileReader:
    @staticmethod
    def read_text_from_path(file_path: Path) -> Tuple[str, bool]:
        """Read text from file with better error handling"""
        try:
            if file_path.suffix.lower() == ".txt":
                text = file_path.read_text(encoding="utf-8", errors="ignore")
                return text, True
                
            elif file_path.suffix.lower() == ".pdf":
                return FileReader._read_pdf(file_path)
                
            elif file_path.suffix.lower() == ".docx":
                return FileReader._read_docx(file_path)
                
            else:
                return f"Unsupported file format: {file_path.suffix}", False
                
        except Exception as e:
            return f"Error reading file: {str(e)}", False

    @staticmethod
    def _read_pdf(file_path: Path) -> Tuple[str, bool]:
        try:
            import PyPDF2
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    page_text = page.extract_text() or ""
                    text += page_text + "\n"
                return text, True
        except ImportError:
            return "Error: PyPDF2 library not installed. Please install: pip install PyPDF2", False
        except Exception as e:
            return f"Error reading PDF: {str(e)}", False

    @staticmethod
    def _read_docx(file_path: Path) -> Tuple[str, bool]:
        try:
            import docx
            doc = docx.Document(str(file_path))
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text, True
        except ImportError:
            return "Error: python-docx library not installed. Please install: pip install python-docx", False
        except Exception as e:
            return f"Error reading DOCX: {str(e)}", False


# =========================
# Enhanced Tkinter UI
# =========================
class EnhancedResumeUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Enhanced Resume Extractor v2.0")
        self.root.geometry("1200x900")
        self.root.configure(bg='#f0f0f0')
        
        # Set icon and style
        self.setup_styles()
        
        self.extractor = EnhancedResumeExtractor()
        self.resumes = []
        self.results = {}
        
        self.create_widgets()
        
    def setup_styles(self):
        """Configure ttk styles for better appearance"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure custom styles
        style.configure('Title.TLabel', font=('Arial', 18, 'bold'), background='#f0f0f0')
        style.configure('Header.TLabel', font=('Arial', 12, 'bold'), background='#f0f0f0')
        style.configure('Success.TButton', background='#4CAF50', foreground='white')
        style.configure('Primary.TButton', background='#2196F3', foreground='white')
        style.configure('Warning.TButton', background='#FF9800', foreground='white')

    def create_widgets(self):
        """Create and arrange UI widgets"""
        # Main container with padding
        main_container = ttk.Frame(self.root, padding="20")
        main_container.pack(fill="both", expand=True)
        
        # Title
        title_label = ttk.Label(main_container, text="🚀 Enhanced Resume Extractor", 
                               style='Title.TLabel')
        title_label.pack(pady=(0, 20))
        
        # Control panel
        self.create_control_panel(main_container)

        # Job Description input area
        jd_frame = ttk.Frame(main_container)
        jd_frame.pack(fill="x", pady=(10, 10))
        ttk.Label(jd_frame, text="Job Description (paste or type):", style='Header.TLabel').pack(anchor="w")
        self.jd_text = scrolledtext.ScrolledText(jd_frame, height=6, width=120, font=("Arial", 10))
        self.jd_text.pack(fill="x", pady=(5, 0))

        # Progress bar
        self.progress = ttk.Progressbar(main_container, mode='determinate')
        self.progress.pack(fill="x", pady=(10, 5))
        
        # Status label
        self.status_label = ttk.Label(main_container, text="Ready to extract resume data")
        self.status_label.pack(pady=(0, 15))
        
        # Results notebook (tabs for different views)
        self.create_results_notebook(main_container)
        
        # Bottom buttons
        self.create_bottom_buttons(main_container)

    def create_control_panel(self, parent):
        """Create the control panel with buttons"""
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill="x", pady=(0, 10))

        # Upload button
        self.upload_btn = ttk.Button(control_frame, text="📂 Upload Resumes",
                                    command=self.upload_resumes, width=20)
        self.upload_btn.pack(side="left", padx=(0, 10))

        # Extract button
        self.extract_btn = ttk.Button(control_frame, text="🔍 Extract Data",
                                     command=self.run_extraction, width=20,
                                     style='Success.TButton')
        self.extract_btn.pack(side="left", padx=(0, 10))

        # Filter Result button (new) - placed next to Extract Data
        self.filter_result_btn = ttk.Button(control_frame, text="🧾 Filter Result",
                                            command=self.filter_results, width=15,
                                            style='Primary.TButton')
        self.filter_result_btn.pack(side="left", padx=(0, 10))

        # Clear button
        self.clear_btn = ttk.Button(control_frame, text="🗑️ Clear Results",
                                     command=self.clear_results, width=15)
        self.clear_btn.pack(side="right")

    def create_results_notebook(self, parent):
        """Create tabbed interface for results"""
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill="both", expand=True, pady=(0, 15))
        
        # Skills from Resume tab
        resume_json_frame = ttk.Frame(self.notebook)
        self.notebook.add(resume_json_frame, text="🧾 Skills from Resume")

        ttk.Label(resume_json_frame, text="Skills from Resume:",
                 style='Header.TLabel').pack(anchor="w", pady=(10, 5))
        self.resume_json_text = scrolledtext.ScrolledText(resume_json_frame, height=20, width=100,
                                                         font=("Consolas", 10))
        self.resume_json_text.pack(fill="both", expand=True, padx=10, pady=5)

        # Skills from JD tab
        jd_json_frame = ttk.Frame(self.notebook)
        self.notebook.add(jd_json_frame, text="📄 Skills from JD")

        ttk.Label(jd_json_frame, text="Skills from JD:",
                 style='Header.TLabel').pack(anchor="w", pady=(10, 5))
        self.jd_json_text = scrolledtext.ScrolledText(jd_json_frame, height=20, width=100,
                                                     font=("Consolas", 10))
        self.jd_json_text.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Summary tab
        summary_frame = ttk.Frame(self.notebook)
        self.notebook.add(summary_frame, text="📈 Summary")
        
        ttk.Label(summary_frame, text="Extraction Summary:", 
                 style='Header.TLabel').pack(anchor="w", pady=(10, 5))
        
        self.summary_text = scrolledtext.ScrolledText(summary_frame, height=20, width=100,
                                                     font=("Arial", 10))
        self.summary_text.pack(fill="both", expand=True, padx=10, pady=5)

    def create_bottom_buttons(self, parent):
        """Create bottom action buttons"""
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill="x")
        
        # Save JSON button
        self.save_json_btn = ttk.Button(button_frame, text="💾 Save JSON", 
                                       command=self.save_json_results, width=15,
                                       style='Primary.TButton')
        self.save_json_btn.pack(side="left", padx=(0, 10))
        
        # Export CSV button
        self.export_csv_btn = ttk.Button(button_frame, text="📋 Export CSV", 
                                        command=self.export_csv, width=15)
        self.export_csv_btn.pack(side="left", padx=(0, 10))
        
        # About button
        self.about_btn = ttk.Button(button_frame, text="ℹ️ About", 
                                   command=self.show_about, width=10)
        self.about_btn.pack(side="right")

    def upload_resumes(self):
        """Handle resume file uploads"""
        file = filedialog.askopenfilename(
            title="Select a Resume File",
            filetypes=[
                ("All Supported", "*.pdf *.docx *.txt"),
                ("PDF files", "*.pdf"),
                ("Word files", "*.docx"), 
                ("Text files", "*.txt"),
                ("All files", "*.*")
            ]
        )

        if file:
            self.resumes = [Path(file)]
            self.status_label.config(text="✅ 1 resume selected")
            messagebox.showinfo("File Selected", "Selected 1 resume file for processing.")
            logger.info(f"Uploaded 1 resume file: {file}")
    def filter_results(self):
        """Compare stored resume vs JD and call LLM (or short-circuit on mandatory failures).

        Uses previously-extracted values in self.results. Short-circuits and writes
        a deterministic low-score when a mandatory certification or mandatory
        experience requirement is not met, otherwise asks the LLM for a score.
        """
        results = self.results or {}
        jd_meta = results.get('_job_description', {})
        jd_skills = jd_meta.get('skills', [])
        jd_certs = jd_meta.get('certifications', [])
        jd_is_mandatory = jd_meta.get('isCertMandatory', 'Optional')
        jd_required_years = jd_meta.get('required_experience_years', None)
        jd_experience_mandatory = jd_meta.get('isExperienceMandatory', 'Optional')

        resume = results.get('resume', {}) or {}

        # Deduplicate resume skills/certs case-insensitively while preserving order
        deduped_resume_skills = []
        seen_skills = set()
        for s in resume.get('skills', []) or []:
            ss = str(s).strip()
            key = ss.lower()
            if ss and key not in seen_skills:
                seen_skills.add(key)
                deduped_resume_skills.append(ss)

        resume_certs = []
        seen_certs = set()
        for c in resume.get('certifications', []) or []:
            cc = str(c).strip()
            key = cc.lower()
            if cc and key not in seen_certs:
                seen_certs.add(key)
                resume_certs.append(cc)

        # include resume overall years
        resume_years = resume.get('overall_experience_years', None)
        # mirror naming: expose JD years in a variable named like resume_years
        jd_years = jd_required_years

        # Enforce missing MUST-HAVE skills: if JD lists must_have items and any are
        # not found in the resume skills (case-insensitive), short-circuit with low match.
        jd_must_have = jd_meta.get('must_have', []) or []
        jd_good_to_have = jd_meta.get('good_to_have', []) or []

        # Do not deterministically short-circuit on missing must-haves here.
        # We will send the JD must-have list to the LLM and let it decide which are missing
        # and what match percentage to return.

        # Deterministic experience enforcement: if JD marks experience as mandatory
        # and the resume's years don't satisfy the JD requirement, short-circuit
        # with an Insufficient Experience result (no LLM call).
        if jd_experience_mandatory == 'yes' and jd_required_years and resume_years is not None:
            req = jd_required_years
            insufficient = False
            try:
                ry = float(resume_years)
            except Exception:
                ry = None

            if ry is not None:
                # Apply a tolerance of +/- 0.5 years around requirements
                tol = 0.5
                if isinstance(req, dict):
                    # exact requirement => allow within +/- tol
                    if 'exact' in req:
                        val = float(req['exact'])
                        if not (val - tol <= ry <= val + tol):
                            insufficient = True
                    else:
                        # closed range -> allow expanded range by tol on both sides
                        if 'min' in req and 'max' in req:
                            minv = float(req['min']) - tol
                            maxv = float(req['max']) + tol
                            if ry < minv or ry > maxv:
                                insufficient = True
                        else:
                            if 'min' in req:
                                if ry < float(req['min']) - tol:
                                    insufficient = True
                            if 'max' in req:
                                if ry > float(req['max']) + tol:
                                    insufficient = True
                else:
                    # numeric scalar: treat as exact with tolerance (not just minimum)
                    try:
                        val = float(req)
                        if not (val - tol <= ry <= val + tol):
                            insufficient = True
                    except Exception:
                        pass

            if insufficient:
                out = {'match_percentage': 10, 'reason': 'Insufficient Experience'}
                pretty = json.dumps(out, indent=2, ensure_ascii=False)
                if hasattr(self, 'summary_text'):
                    self.summary_text.delete('1.0', tk.END)
                    self.summary_text.insert(tk.END, pretty)
                return

        # Configure API key (original hard-coded string kept for compatibility)
        try:
            genai.configure(api_key="xyz")
        except Exception:
            pass
        model = genai.GenerativeModel("gemini-pro-latest")

        # Build succinct prompt including experience fields and MUST-HAVE context
        prompt = f"""
                You are a recruiter. Compare the candidate’s resume to the job description,
                focusing ONLY on skills, certifications, and years of experience.

                Resume:
                - Skills: {deduped_resume_skills}
                - Certifications: {resume_certs}
                - Total Experience (years): {resume_years}

                Job Description:
                - Skills: {jd_skills}
                - Certifications: {jd_certs}
                - Is Certification Mandatory: {jd_is_mandatory}
                - Required Experience (years): {jd_required_years}
                - Is Experience Mandatory: {jd_experience_mandatory}

                CRITICAL: These are MANDATORY must-have skills: {jd_must_have}

                IMPORTANT NOTES:
                - Skills lists may contain noise, descriptive text, and extra words
                - Focus on extracting core technology/skill names from the text
                - Must-have skills may include phrases like "Front-end React js" (extract: React), ".Net Core and MVC" (extract: .NET Core, MVC)
                - Resume skills may also contain descriptive text - extract the core technologies
                - Match skills intelligently based on technology names, not exact string matches
                - Consider skill variations (React = React.js = ReactJS, .NET = DotNet = Net Core, etc.)
                - Reason should be in 5-6 words, concise and to the point

                STRICT ENFORCEMENT RULES:
                1. Extract core technology names from both resume skills and must-have lists
                2. IF ANY technology from must-have list is missing from resume, IMMEDIATELY return 15-25%
                3. Use intelligent matching - don't require exact string matches
                4. Examples: "Front-end React js" contains React, "agile methodology" contains Agile
                5. If ALL must-have technologies are found in resume, evaluate normally
                6. If mandatory certification missing, return percentage 10-30%
                7. If experience doesn't match requirements, return percentage 10-30%

                REASON FORMAT:
                - If missing must-haves: "Poor match. Missing mandatory skills: [list missing items]. Found: [list found items]"
                - If good match: "Strong match. Key skills: [list 3-4 matching skills]"

                Return ONLY valid JSON: {{"match_percentage": number, "reason": "detailed explanation", "missing_must_have": ["list"]}}
                """

        try:
            response = model.generate_content(prompt)

            response_text = getattr(response, 'text', None)
            if not response_text and hasattr(response, 'output'):
                response_text = str(response.output)
            elif not response_text and hasattr(response, 'candidates'):
                response_text = '\n'.join(getattr(c, 'text', str(c)) for c in response.candidates)
            if not response_text:
                response_text = str(response)

            try:
                parsed = json.loads(response_text)
                pretty = json.dumps(parsed, indent=2, ensure_ascii=False)
            except Exception:
                pretty = response_text

            if hasattr(self, 'summary_text'):
                self.summary_text.delete('1.0', tk.END)
                self.summary_text.insert(tk.END, pretty)

        except Exception as e:
            logger.error(f'LLM call failed: {e}')
            messagebox.showerror('Filter Error', f'Failed to compute match using LLM: {e}')
    def run_extraction(self):
        """Run the extraction process with progress tracking"""
        if not self.resumes:
            messagebox.showerror("No Files", "Please upload resume files first.")
            return
        
        # Disable buttons during processing
        self.set_buttons_state('disabled')
        self.progress['value'] = 0
        self.progress['maximum'] = len(self.resumes)
        self.status_label.config(text="🔄 Extracting data... Please wait")
        
        import threading
        
        def extraction_worker():
            try:
                results = {}
                jd_text = self.jd_text.get("1.0", tk.END).strip()
                jd_skills, jd_certs = self.extractor.extract_skills_and_certs(jd_text) if jd_text else ([], [])
                jd_years_required, jd_is_experience_mandatory = self.extractor.extract_jd_experience_requirement(jd_text)
                jd_cert_mandatory = self.extractor._jd_cert_mandatory(jd_text, jd_certs)
                results['_job_description'] = {
                    'text': jd_text,
                    'skills': jd_skills,
                    'certifications': jd_certs,
                    'isCertMandatory': jd_cert_mandatory,
                    'required_experience_years': jd_years_required,
                    'isExperienceMandatory': jd_is_experience_mandatory,
                    # extract explicit must-have / good-to-have sections
                    'must_have': self.extractor.extract_jd_sections(jd_text)[0],
                    'good_to_have': self.extractor.extract_jd_sections(jd_text)[1]
                }
                # Process only first resume if present
                if self.resumes:
                    resume_path = self.resumes[0]
                    text, success = FileReader.read_text_from_path(resume_path)
                    if success:
                        data = self.extractor.extract(text)
                        results['resume'] = data.to_dict()
                    else:
                        results['resume'] = {}
                
                # Update UI with results
                self.root.after(0, lambda: self.display_results(results))
                
            except Exception as e:
                self.root.after(0, lambda: self.handle_extraction_error(str(e)))
        
        threading.Thread(target=extraction_worker, daemon=True).start()

    def update_progress(self, current):
        """Update progress bar"""
        self.progress['value'] = current + 1
        self.status_label.config(text=f"🔄 Processing file {current + 1} of {len(self.resumes)}...")
        self.root.update_idletasks()

    def display_results(self, results):
        """Display extraction results in UI"""
        self.results = results
        jd_meta = results.get('_job_description', {'text': '', 'skills': []})
        resume_data = results.get('resume', {})
        # Update Resume JSON pane
        self.resume_json_text.delete("1.0", tk.END)
        formatted_resume_json = json.dumps({'skills': resume_data.get('skills', []), 'certifications': resume_data.get('certifications', [])}, indent=2, ensure_ascii=False)
        self.resume_json_text.insert(tk.END, formatted_resume_json)
        # Update JD JSON pane
        self.jd_json_text.delete("1.0", tk.END)
        formatted_jd_json = json.dumps({
            'skills': jd_meta.get('skills', []),
            'certifications': jd_meta.get('certifications', []),
            'isCertMandatory': jd_meta.get('isCertMandatory', 'Optional'),
            'must_have': jd_meta.get('must_have', []),
            'good_to_have': jd_meta.get('good_to_have', [])
        }, indent=2, ensure_ascii=False)
        self.jd_json_text.insert(tk.END, formatted_jd_json)
        # Update summary tab
        self.update_summary_tab(results)
        self.status_label.config(text="✅ Extraction completed")
        self.progress['value'] = self.progress['maximum']
        self.set_buttons_state('normal')
        logger.info("Extraction completed for 1 resume")

    def update_summary_tab(self, results):
        """Generate and display summary statistics"""
        self.summary_text.delete("1.0", tk.END)
        
        total_files = len(results)
        successful = sum(1 for r in results.values() if "error" not in r)
        failed = total_files - successful
        
        # Calculate average confidence score
        confidence_scores = [r.get('confidence_score', 0) for r in results.values() if "error" not in r]
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        # Generate summary text
        summary = f"""EXTRACTION SUMMARY
{'=' * 50}

📁 Total Files Processed: {total_files}
✅ Successfully Extracted: {successful}
❌ Failed Extractions: {failed}
📊 Average Confidence Score: {avg_confidence:.1f}%

DETAILED BREAKDOWN:
{'=' * 50}

"""
        
        # Add individual file results
        for filename, data in results.items():
            if "error" in data:
                summary += f"❌ {filename}\n"
                summary += f"   Error: {data['error']}\n\n"
            else:
                confidence = data.get('confidence_score', 0)
                summary += f"✅ {filename} (Confidence: {confidence}%)\n"
                
                # Add extracted data summary
                if data.get('name'):
                    summary += f"   👤 Name: {data['name']}\n"
                if data.get('email'):
                    summary += f"   📧 Email: {data['email']}\n"
                if data.get('phone'):
                    summary += f"   📞 Phone: {data['phone']}\n"
                if data.get('overall_experience_years'):
                    summary += f"   💼 Experience: {data['overall_experience_years']} years\n"
                if data.get('skills'):
                    summary += f"   🎯 Skills: {len(data['skills'])} found\n"
                # technologies display removed
                if data.get('experience'):
                    summary += f"   🏢 Work Experience: {len(data['experience'])} entries\n"
                if data.get('education'):
                    summary += f"   🎓 Education: {len(data['education'])} entries\n"
                
                summary += "\n"
        
        # Add skills statistics
        all_skills = []
        for data in results.values():
            if "error" not in data:
                all_skills.extend(data.get('skills', []))
        
        if all_skills:
            summary += f"\nTOP SKILLS ACROSS ALL RESUMES:\n{'=' * 50}\n"
            from collections import Counter
            skill_counts = Counter(all_skills)
            summary += "\n🎯 Most Common Skills:\n"
            for skill, count in skill_counts.most_common(10):
                summary += f"   • {skill}: {count} resume(s)\n"
        
        self.summary_text.insert(tk.END, summary)

    def handle_extraction_error(self, error_msg):
        """Handle extraction errors"""
        self.status_label.config(text="❌ Extraction failed")
        self.set_buttons_state('normal')
        messagebox.showerror("Extraction Error", f"Extraction failed: {error_msg}")
        logger.error(f"Extraction failed: {error_msg}")

    def set_buttons_state(self, state):
        """Enable or disable buttons"""
        buttons = [self.upload_btn, self.extract_btn, getattr(self, 'filter_result_btn', None), self.clear_btn,
                  self.save_json_btn, self.export_csv_btn]
        for btn in buttons:
            try:
                if btn is not None:
                    btn.config(state=state)
            except Exception:
                pass

    def clear_results(self):
        """Clear all results and reset UI"""
        if messagebox.askyesno("Clear Results", "Are you sure you want to clear all results?"):
            self.results = {}
            self.resumes = []
            self.resume_json_text.delete("1.0", tk.END)
            try:
                self.jd_text.delete("1.0", tk.END)
            except Exception:
                pass
            try:
                self.jd_json_text.delete("1.0", tk.END)
            except Exception:
                pass
            self.summary_text.delete("1.0", tk.END)
            self.progress['value'] = 0
            self.status_label.config(text="Ready to extract resume data")
            
            # Add welcome message
            welcome_msg = """Welcome to Enhanced Resume Extractor v2.0!

🚀 FEATURES:
• Extract contact information (name, email, phone)
• Identify skills
• Parse work experience and education
• Professional certifications detection
• Language skills identification
• Confidence scoring for extraction quality
• Enhanced data cleaning and validation

📝 INSTRUCTIONS:
1. Click '📂 Upload Resumes' to select files (PDF, DOCX, TXT)
2. Click '🔍 Extract Data' to process the resumes
3. View results in JSON format or Summary tabs
4. Save results as JSON or export to CSV

⚙️ REQUIREMENTS:
• For PDF files: pip install PyPDF2
• For DOCX files: pip install python-docx

Ready to process your resume files!"""
            
            self.resume_json_text.insert(tk.END, welcome_msg)

    def save_json_results(self):
        """Save results to JSON file"""
        if not self.results:
            messagebox.showwarning("No Results", "No results to save. Please run extraction first.")
            return
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_filename = f"resume_extraction_results_{timestamp}.json"
            
            file_path = filedialog.asksaveasfilename(
                defaultextension=".json",
                initialvalue=default_filename,
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                title="Save JSON Results"
            )
            
            if file_path:
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(self.results, f, indent=2, ensure_ascii=False)
                messagebox.showinfo("Success", f"Results saved successfully to:\n{file_path}")
                logger.info(f"Results saved to {file_path}")
                
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save results: {str(e)}")
            logger.error(f"Failed to save results: {str(e)}")

    def export_csv(self):
        """Export results to CSV format"""
        if not self.results:
            messagebox.showwarning("No Results", "No results to export. Please run extraction first.")
            return
        
        try:
            import csv
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_filename = f"resume_extraction_results_{timestamp}.csv"
            
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                initialvalue=default_filename,
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                title="Export to CSV"
            )
            
            if file_path:
                with open(file_path, "w", newline="", encoding="utf-8") as csvfile:
                    # Define CSV headers
                    fieldnames = [
                            'filename', 'name', 'email', 'phone', 'overall_experience_years',
                            'confidence_score', 'skills', 'experience', 
                            'education', 'certifications', 'languages', 'error'
                        ]
                    
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    for filename, data in self.results.items():
                        row = {'filename': filename}
                        
                        if "error" in data:
                            row['error'] = data['error']
                        else:
                            for field in fieldnames[1:-1]:  # Skip filename and error
                                value = data.get(field, '')
                                # Convert lists to comma-separated strings
                                if isinstance(value, list):
                                    row[field] = '; '.join(str(item) for item in value)
                                else:
                                    row[field] = value or ''
                        
                        writer.writerow(row)
                
                messagebox.showinfo("Success", f"Results exported successfully to:\n{file_path}")
                logger.info(f"Results exported to CSV: {file_path}")
                
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export to CSV: {str(e)}")
            logger.error(f"Failed to export to CSV: {str(e)}")

    def show_about(self):
        """Show about dialog"""
        about_text = """Enhanced Resume Extractor v2.0

🚀 A powerful tool for extracting structured data from resumes

✨ FEATURES:
• Contact information extraction
• Skills identification
• Work experience parsing
• Education history extraction
• Professional certifications detection
• Language skills identification
• Confidence scoring
• Multiple export formats (JSON, CSV)

👨‍💻 DEVELOPER:
Created with Python, Tkinter, and advanced regex patterns

📋 SUPPORTED FORMATS:
• PDF files (requires PyPDF2)
• Word documents (requires python-docx)
• Text files

⚡ PERFORMANCE:
• Multi-threaded processing
• Progress tracking
• Error handling and recovery

For updates and support, please check the documentation."""

        messagebox.showinfo("About Enhanced Resume Extractor", about_text)


# =========================
# Main Application Entry Point
# =========================
def main():
    """Main application entry point"""
    try:
        # Create and configure root window
        root = tk.Tk()
        
        # Set window icon (if available)
        try:
            # You can add an icon file here if available
            # root.iconbitmap('resume_icon.ico')
            pass
        except:
            pass
        
        # Create application instance
        app = EnhancedResumeUI(root)
        
        # Add welcome message
        app.clear_results()
        
        # Start the application
        logger.info("Enhanced Resume Extractor started")
        root.mainloop()
        
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        messagebox.showerror("Startup Error", f"Failed to start application: {str(e)}")


if __name__ == "__main__":
    main()
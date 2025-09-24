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
                "vue", "node.js", "django", "flask", "spring", "dotnet", ".net", "sql", 
                "mysql", "postgresql", "mongodb", "oracle", "aws", "azure", "gcp", "docker", 
                "kubernetes", "git", "jenkins", "linux", "windows", "html", "css", "sass",
                "power bi", "tableau", "spark", "hadoop", "pandas", "numpy", "scikit-learn", 
                "tensorflow", "pytorch", "rest api", "graphql", "selenium", "jira", 
                "confluence", "matlab", "sas", "r", "php", "swift", "go", "rust", "scala",
                "bash", "shell", "redis", "elasticsearch", "firebase", "android", "ios", 
                "xcode", "visual studio", "unity", "unreal", "salesforce", "sap", "abap",
                "powerapps", "servicenow", "bigquery", "looker", "airflow", "terraform",
                "excel", "machine learning", "data analysis", "project management", "leadership",
                "communication", "problem solving", "teamwork", "agile", "scrum",
                "database design", "system architecture", "devops", "ci/cd"
            ]

            # Convert to dict so SkillExtractor can use it
        return {skill: "TECH" for skill in set(skills)}


    def _load_certification_keywords(self) -> List[str]:
        return [
            'aws certified', 'azure certified', 'google cloud certified', 'cissp', 
            'cism', 'pmp', 'scrum master', 'six sigma', 'itil', 'comptia', 'ccna', 
            'ccnp', 'ccie', 'mcse', 'oracle certified', 'salesforce certified',
            'certified kubernetes', 'docker certified', 'red hat certified'
        ]

    def _load_language_keywords(self) -> List[str]:
        return [
            'english', 'spanish', 'french', 'german', 'italian', 'portuguese', 
            'chinese', 'japanese', 'korean', 'russian', 'arabic', 'hindi', 
            'dutch', 'swedish', 'norwegian', 'danish'
        ]

    def extract(self, text: str) -> ResumeData:
        """Main extraction method with improved error handling"""
        try:
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
            
            # Calculate confidence score
            data.calculate_confidence()
            
            return data
            
        except Exception as e:
            logger.error(f"Error during extraction: {str(e)}")
            return ResumeData()

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
        # Use SkillNER if available
        if getattr(self, 'skillner', None) is not None:
            try:
                skills_out = []

                skills_dict = self._load_tech_keywords()

                matcher = PhraseMatcher(NLP.vocab, attr="LOWER")
                matcher.add("CUSTOM_SKILLS", [NLP.make_doc(skill) for skill in skills_dict.keys()])
                doc = NLP(text)
                skills_out = {doc[start:end].text for _, start, end in matcher(doc)}

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

                return unique[:15]
            except Exception as e:
                logger.warning(f"SkillNER extraction failed ({e}); falling back to keyword extraction")

        # Fallback logic: original keyword + regex extraction
        found_skills = []
        text_lower = text.lower()

        for skill in self.skill_keywords:
            if re.search(rf'\b{re.escape(skill.lower())}\b', text_lower):
                found_skills.append(skill.title())

        # Look for skills section
        skills_match = re.search(r'(skills?|competencies|expertise)([\s\S]{0,300})', text, re.IGNORECASE)
        if skills_match:
            skills_section = skills_match.group(2)
            additional_skills = re.findall(r'[•\-\*]?\s*([A-Za-z][A-Za-z\s&/\-]{2,20})', skills_section)
            found_skills.extend([skill.strip().title() for skill in additional_skills if skill.strip()])

        # Remove duplicates while preserving order
        seen = set()
        unique_skills = []
        for skill in found_skills:
            if skill.lower() not in seen:
                seen.add(skill.lower())
                unique_skills.append(skill)

        return unique_skills[:15]

    

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
        
        return list(set(found_certs))[:10]

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
        files = filedialog.askopenfilenames(
            title="Select Resume Files",
            filetypes=[
                ("All Supported", "*.pdf *.docx *.txt"),
                ("PDF files", "*.pdf"),
                ("Word files", "*.docx"), 
                ("Text files", "*.txt"),
                ("All files", "*.*")
            ]
        )
        
        if files:
            self.resumes = [Path(f) for f in files]
            self.status_label.config(text=f"✅ {len(self.resumes)} resume(s) selected")
            messagebox.showinfo("Files Selected", 
                              f"Selected {len(self.resumes)} resume files for processing.")
            logger.info(f"Uploaded {len(self.resumes)} resume files")

    def filter_results(self):
        results = self.results or {}

        # --- Extract JD skills ---
        jd_skills = []
        jd_meta = results.get('_job_description')
        if isinstance(jd_meta, dict):
            jd_skills = jd_meta.get('skills', []) or []

        # --- Extract resume skills (excluding JD) ---
        resume_skills = []
        for k, data in results.items():
            if k != '_job_description' and isinstance(data, dict):
                resume_skills.extend(data.get('skills', []) or [])

        # --- Deduplicate resume skills ---
        seen, deduped_resume_skills = set(), []
        for skill in resume_skills:
            skill_str = str(skill).strip()
            if skill_str and skill_str.lower() not in seen:
                seen.add(skill_str.lower())
                deduped_resume_skills.append(skill_str)

        # --- Configure Gemini ---
        try:
            genai.configure(api_key="xyz")
        except Exception:
            pass  # Ignore offline / test failures

        model = genai.GenerativeModel("gemini-1.5-flash")

        # --- Prompt ---
        prompt = f"""
        You are a recruiter. Compare the candidate's skills to the job description skills.

Resume Skills: {deduped_resume_skills}
Job Description Skills: {jd_skills}

Rules:
1. Calculate overall matching percentage (0-100) based on skills that are the same or highly similar in meaning.
   - For example, "React" and "ReactJS" or "AWS" and "Amazon Web Services" should be considered matches.
2. If percentage > 75, reason = "Matching: <list 2-3 key matched JD skills>".
3. If percentage < 40, reason = "Missing: <list 2-3 key missing JD skills>".
4. Otherwise, reason = "Partial Match: <list 2-3 matched/missing skills>".
5. Output only valid JSON with keys: match_percentage, reason. No extra text.
6. Use your knowledge and semantic reasoning to identify matches, not just literal string equality.
        """

        try:
            # Call generate_content using the SDK's supported signature (prompt only).
            # Previously supplied kwargs (temperature, top_p, candidate_count) are
            # not supported by this client wrapper and caused a TypeError.
            response = model.generate_content(prompt, generation_config={"temperature": 0} )

            # --- Extract response text robustly ---
            response_text = getattr(response, "text", None)

            if not response_text and hasattr(response, "output"):
                response_text = str(response.output)
            elif not response_text and hasattr(response, "candidates"):
                response_text = "\n".join(
                    getattr(c, "text", str(c)) for c in response.candidates
                )
            if not response_text:
                response_text = str(response)

            # --- Parse as JSON if possible ---
            try:
                parsed = json.loads(response_text)
                pretty = json.dumps(parsed, indent=2, ensure_ascii=False)
            except Exception:
                pretty = response_text

            # --- Display in Summary tab ---
            if hasattr(self, "summary_text"):
                self.summary_text.delete("1.0", tk.END)
                self.summary_text.insert(tk.END, pretty)

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            messagebox.showerror("Filter Error", f"Failed to compute match using LLM: {e}")


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
                # Extract Job Description skills (from the JD input area)
                try:
                    jd_text = self.jd_text.get("1.0", tk.END).strip()
                    jd_skills = self.extractor._extract_skills(jd_text) if jd_text else []
                    results['_job_description'] = {'text': jd_text, 'skills': jd_skills}
                except Exception as e:
                    logger.warning(f"Failed to extract JD skills: {e}")
                    results['_job_description'] = {'text': '', 'skills': []}
                for i, resume_path in enumerate(self.resumes):
                    # Update progress
                    self.root.after(0, lambda i=i: self.update_progress(i))
                    
                    try:
                        text, success = FileReader.read_text_from_path(resume_path)
                        if success:
                            data = self.extractor.extract(text)
                            results[resume_path.name] = data.to_dict()
                        else:
                            results[resume_path.name] = {"error": text, "confidence_score": 0.0}
                    except Exception as e:
                        results[resume_path.name] = {"error": f"Processing failed: {str(e)}", 
                                                   "confidence_score": 0.0}
                        logger.error(f"Error processing {resume_path.name}: {str(e)}")
                
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
        # Prepare resume-only entries and JD metadata
        jd_meta = results.get('_job_description', {'text': '', 'skills': []})
        resume_entries = {k: v for k, v in results.items() if k != '_job_description'}

        # Update Resume JSON (skills) pane
        try:
                self.resume_json_text.delete("1.0", tk.END)
                # Show only skills per resume for the 'Skills from Resume' tab
                resume_skills_display = {fn: {'skills': data.get('skills', [])} if isinstance(data, dict) else {} for fn, data in resume_entries.items()}
                # Do not include JD skills here; this pane must show only resume skills
                formatted_resume_json = json.dumps(resume_skills_display, indent=2, ensure_ascii=False)
                self.resume_json_text.insert(tk.END, formatted_resume_json)
        except Exception:
            pass

        # Update JD JSON (skills) pane — show only the skills list for clarity
        try:
            self.jd_json_text.delete("1.0", tk.END)
            jd_skills_only = jd_meta.get('skills', []) if isinstance(jd_meta, dict) else []
            formatted_jd_json = json.dumps({'skills': jd_skills_only}, indent=2, ensure_ascii=False)
            self.jd_json_text.insert(tk.END, formatted_jd_json)
        except Exception:
            pass
        
        # Update summary tab
        self.update_summary_tab(results)
        
        # Update status
        successful = sum(1 for r in results.values() if "error" not in r)
        self.status_label.config(text=f"✅ Extraction completed: {successful}/{len(results)} successful")
        self.progress['value'] = self.progress['maximum']
        
        # Re-enable buttons
        self.set_buttons_state('normal')
        
        logger.info(f"Extraction completed for {len(results)} files")

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
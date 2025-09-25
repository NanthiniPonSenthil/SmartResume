import google.generativeai as genai

# Initialize Google API key
API_KEY = "XYZ" 
genai.configure(api_key=API_KEY)

# Example skills
resume_skills = ["Python", "SQL", "AWS", '.NET']
jd_skills = ["Python", "SQL","Docker","MySQL","Kubernetes","Java","Azure","React"]

# Model
model = genai.GenerativeModel("gemini-pro-latest")

# Prompt for Google Gemini
prompt = f"""
You are a recruiter. Compare the candidate's skills to the job description skills.

Resume Skills: {resume_skills}
Job Description Skills: {jd_skills}

Rules:
1. Calculate overall matching percentage (0-100).
   Formula: (matched_skills / total_jd_skills) * 100.
2. If percentage > 75, reason = "Matching: <list 2-3 key matched JD skills>".
3. If percentage < 40, reason = "Missing: <list 2-3 key missing JD skills>".
4. Otherwise, reason = "Partial Match: <list 2-3 matched/missing skills>".
5. Output only valid JSON with keys: match_percentage, reason. No extra text.
"""


# Generate output
response = model.generate_content(prompt)

# Print the output text
print(response.text)

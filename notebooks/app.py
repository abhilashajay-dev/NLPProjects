
import spacy
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the spaCy language model
nlp = spacy.load("en_core_web_md")

# Function to extract named entities (skills, experience, education, projects)
def extract_entities(text):
    lines = text.split('\n')
    skills = []
    experience = []
    education = []
    projects = []

    for line in lines:
        parts = line.strip().split(':')
        if len(parts) == 2:
            entity_type, entity_text = parts[0].strip(), parts[1].strip()
            if entity_type == "SKILL":
                skills.append(entity_text)
            elif entity_type == "EXPERIENCE":
                experience.append(entity_text)
            elif entity_type == "EDUCATION":
                education.append(entity_text)
            elif entity_type == "PROJECT":
                projects.append(entity_text)

    return skills, experience, education, projects

# Streamlit UI
st.title("Resume Ranking App")

# User input for job description
job_description = st.text_area("Enter the job description:")

# User input for resumes
st.write("Enter the resumes in entity format. Use the following format:")
st.write("ENTITY_TYPE: ENTITY_TEXT")
st.write("Example:")
st.write("SKILL: Python, machine learning\nEXPERIENCE: Worked on various projects, including a recommendation system.\n")
resumes_text = st.text_area("Enter the resumes (one per line):")

if job_description and resumes_text:
    # Combine the resumes and the job description into one list
    resumes = resumes_text.strip().split('\n')
    all_text = resumes + [job_description]

    # Create a CountVectorizer to convert text to a matrix of token counts
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(all_text)

    # Calculate cosine similarity between the job description and all resumes
    cosine_similarities = cosine_similarity(X[-1], X[:-1])

    # Create a list of (resume, similarity score) pairs
    resume_scores = list(zip(resumes, cosine_similarities[0]))

    # Sort the resumes by similarity score in descending order
    resume_scores.sort(key=lambda x: x[1], reverse=True)

    # Display the ranked resumes with named entities
    st.header("Ranked Resumes:")
    for i, (resume, score) in enumerate(resume_scores):
        skills, experience, education, projects = extract_entities(resume)
        st.subheader(f"Rank {i + 1}: Similarity Score = {score:.4f}")
        st.write("Skills:", ", ".join(skills))
        st.write("Experience:", ", ".join(experience))
        st.write("Education:", ", ".join(education))
        st.write("Projects:", ", ".join(projects))
        st.write("Resume:", resume)
        st.write("---")

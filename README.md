

# Resume Matching with Job Description

This project demonstrates how to calculate the similarity between resumes and a job description using natural language processing (NLP) techniques. It leverages TF-IDF (Term Frequency-Inverse Document Frequency) vectorization and cosine similarity to assess how well a resume matches a given job description.

## Features
- Preprocess text data (tokenization, stopwords removal, and punctuation handling).
- Calculate cosine similarity between multiple resumes and a single job description.
- Generate a bar chart visualizing the similarity scores of different resumes.
- Save the results (similarity scores) to a CSV file for further analysis.

## Requirements

Before running the code, make sure you have the following libraries installed:

- `nltk`
- `pandas`
- `matplotlib`
- `scikit-learn`

You can install the required libraries using the following pip command:

```bash
pip install nltk pandas matplotlib scikit-learn
```

## How to Use

1. **Download Necessary NLTK Resources**:  
   The code uses NLTK for tokenization and stopwords removal. The necessary NLTK resources will be automatically downloaded when the script is run.

2. **Input Data**:  
   Provide a list of resume texts and a job description as strings.  
   Each resume text should be a string containing relevant information about a candidate's experience and skills.  
   The job description should also be a string describing the job role and requirements.

3. **Calculate Similarity**:  
   The function `calculate_similarity(resume_texts, jd_text)` processes the resumes and the job description, then computes the cosine similarity between them.

4. **Plot Results**:  
   The function `plot_similarity(scores, resume_names)` generates a horizontal bar chart to visualize the similarity scores.

5. **Save Results**:  
   The function `save_to_csv(resume_names, scores, filename)` saves the similarity scores along with the resume names to a CSV file for further analysis.

## Example

```python
import os
import nltk
import string
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Make sure necessary NLTK resources are downloaded
nltk.download('punkt')

# Function to preprocess text
def preprocess_text(text, remove_stopwords=True):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [word for word in tokens if word not in string.punctuation and not word.isdigit()]
    if remove_stopwords:
        stopwords = nltk.corpus.stopwords.words('english')
        tokens = [word for word in tokens if word not in stopwords]
    return ' '.join(tokens)

# Function to calculate cosine similarity between resumes and job descriptions
def calculate_similarity(resume_texts, jd_text):
    processed_resumes = [preprocess_text(resume) for resume in resume_texts]
    processed_jd = preprocess_text(jd_text)
    
    all_texts = processed_resumes + [processed_jd]
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    
    return cosine_similarities.flatten()

# Function to generate a bar chart of similarity scores
def plot_similarity(scores, resume_names):
    plt.figure(figsize=(10, 6))
    plt.barh(resume_names, scores, color='skyblue')
    plt.xlabel('Cosine Similarity Score')
    plt.title('Top Resume Matches with Job Description')
    plt.gca().invert_yaxis()
    plt.show()

# Function to save results to a CSV
def save_to_csv(resume_names, scores, filename="similarity_scores.csv"):
    df = pd.DataFrame({
        'Resume': resume_names,
        'Similarity Score': scores
    })
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")

# Example data: List of resumes and a job description
resume_texts = [
    "Saurabh Gaikwad, Data Analyst at Bonace Engineers, developed ETL pipelines using Python, SQL, and Pandas.",
    "John Doe, Data Scientist with experience in ML and data visualization, proficient in Python, R, SQL.",
    "Alice Smith, Business Analyst, specializing in process optimization and data analysis in operations."
]
jd_text = "Amazon is looking for a Data Analyst to manage data pipelines, perform analysis, and drive operational improvements."

# Calculate similarity scores
similarity_scores = calculate_similarity(resume_texts, jd_text)

# Plot similarity scores
plot_similarity(similarity_scores, ["Resume 1", "Resume 2", "Resume 3"])

# Save results to CSV
save_to_csv(["Resume 1", "Resume 2", "Resume 3"], similarity_scores)
```

## Output
1. **Similarity Scores**: The cosine similarity between each resume and the job description is calculated.
2. **Visualization**: A bar chart is generated to show which resumes are the closest match to the job description.
3. **CSV File**: The similarity scores are saved to a CSV file, with columns for the resume name and the corresponding similarity score.


![Screenshot 2025-04-12 050313](https://github.com/user-attachments/assets/8cbdadca-8be4-416c-90b9-7ea6ff9b7039)




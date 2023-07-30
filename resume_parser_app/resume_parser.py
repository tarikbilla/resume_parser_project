# pip install pdfminer.six # for python3
# pip install docx2txt
# pip install nltk
# pip install spacy==2.3.5
# python -m spacy download en_core_web_lg
# pip install skillNer


# array manipulation
import numpy as np
import pandas as pd

# systems
import os, io, re
import zipfile
import datetime
import json

# NLP
import spacy
from sklearn.metrics.pairwise import cosine_similarity
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.matcher import PhraseMatcher
from skillNer.general_params import SKILL_DB
from skillNer.skill_extractor_class import SkillExtractor

# PDF Miner
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from pdfminer.pdftypes import resolve1







# NLP
nlp = spacy.load("en_core_web_lg")
allow_stop_words = ["\n\n"]
allow_punct = ["-", "+", "@", ".", "\", ""/", "(", ")"]
for word in allow_stop_words:
  nlp.vocab[word].is_stop = True

for word in allow_punct:
  nlp.vocab[word].is_punct = False

skill_extractor = SkillExtractor(nlp, SKILL_DB, PhraseMatcher)
ngramed_score_threshold = 0.7

# Reg
education_keywords = ["education", "degree", "university", "college"]
degree_patterns = r"\b(bachelor|b\.?a\.?|master|m\.?a\.?|ph\.?d|doctorate|b\.?sc\.?|m\.?sc\.?)\b[^,]*"
phone_pattern = re.compile(r'(\+\d{1,2}\s?)?(\(?\d{3}\)?[\s.-]?)?\d{3}[\s.-]?\d{4}')

# Custom
URL_PRE = "https://"
ALLOWED_FILE_TYPES = [".pdf", ".doc", ".docx"]
UNZIP_TO_FLD = "resumes"






# PDF Miner
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from pdfminer.pdftypes import resolve1

def extract_text_from_pdf(pdf_path):
    extracted_text = ""
    #extracted_hyperlinks = []

    with open(pdf_path, 'rb') as fh:
        for page in PDFPage.get_pages(fh, caching=True, check_extractable=True):
            # creating a resource manager
            resource_manager = PDFResourceManager()
            # creating a file handle
            fake_file_handle = io.StringIO()
            # create a text converter object
            converter = TextConverter(
                                resource_manager,
                                fake_file_handle,
                                codec='utf-8',
                                laparams=LAParams()
                        )
            # create a page interpreter
            page_interpreter = PDFPageInterpreter(
                                resource_manager,
                                converter
                            )
            # process current page
            page_interpreter.process_page(page)
            # extract text
            extracted_text += fake_file_handle.getvalue()

            # # extract hyperlinks
            # if hasattr(page, "annots") and page.annots:
            #     for annotation_ref in page.annots:
            #         annotation = resolve1(annotation_ref)
            #         subtype = str(annotation.get("Subtype"))
            #         if subtype == "/'Link'":
            #             uri_bytes = annotation.get("A").get("URI")
            #             if uri_bytes:
            #                 uri = uri_bytes.decode('utf-8')
            #                 extracted_hyperlinks.append(uri)

            # close all
            converter.close()
            fake_file_handle.close()

            return extracted_text





import docx2txt
def extract_text_from_doc(doc_path):
    temp = docx2txt.process(doc_path)
    text = [line.replace('\t', ' ') for line in temp.split('\n') if line]
    return ' '.join(text)




def preprocess(text):
  doc = nlp(text) # tokenization
  # Lemmatization + Stop words and punctuations removal
  clean_tokens = [token.lemma_ for token in doc if not token.is_punct and not token.is_stop]
  nlp_text = nlp(" ".join(clean_tokens))
  return nlp_text



def extractNER(doc):
  persons = []
  organizations = []
  for entity in doc.ents:
      if entity.label_ == 'PERSON':
          persons.append(entity.text)
      elif entity.label_ == 'ORG':
          organizations.append(entity.text)
  return persons, organizations




def extract_email_addresses(doc):
  # Extract email addresses
  email_addresses = set()
  for token in doc:
    if token.like_email:
      email_addresses.add(token.text)
  return email_addresses





def extract_phones(doc):
  phone_numbers = set()
  for sentence in doc.sents:
    lines = sentence.text.split("\n")
    for line in lines:
      matches = re.finditer(phone_pattern, line.replace(" ", ""))
      for match in matches:
        phone_numbers.add(match.group())

  return phone_numbers






def extractSkills(text):
  annotations = skill_extractor.annotate(text)

  skills = set()
  skills_full_match = annotations["results"]["full_matches"]
  for skill in skills_full_match:
    skills.add((skill["doc_node_value"], skill_extractor.skills_db.get(skill["skill_id"])["skill_name"], skill["score"]))

  skills_ngramed_scored = annotations["results"]["ngram_scored"]
  for skill in skills_ngramed_scored:
    n_gramed_score = skill["score"]/ skill["len"]
    if n_gramed_score > ngramed_score_threshold:
      skills.add((skill["doc_node_value"], skill_extractor.skills_db.get(skill["skill_id"])["skill_name"], n_gramed_score))

  return skills



def extractEducation(doc):
    # Extract Relevant Sentences
    education_sections = []
    for sentence in doc.sents:
        if any(keyword in sentence.text.lower() for keyword in education_keywords):
            education_sections.append(sentence)

    # Perform Named Entity Recognition (NER)
    education_entities = []
    for section in education_sections:
        for entity in section.ents:
            if entity.label_ in ["ORG"]:
                education_entities.append(entity.text)

    # define degree patterns and find a match
    degrees = []

    for section in education_sections:
        sentence_text = section.text  # Convert the Span object to string
        extracted_degrees = re.findall(degree_patterns, sentence_text, flags=re.IGNORECASE)
        for degree in extracted_degrees:
            degrees.append(degree)

    return set(education_entities + degrees)

def extractLinks(doc):
    links = []
    for token in doc:
        if token.like_url:
            link = token.text
            link = URL_PRE + link if not link.startswith(URL_PRE) else link
            links.append(link)
    return links

def resume_parser(file_path):
    out = []
    ext = os.path.splitext(file_path)[1].lower()
    if ext in ALLOWED_FILE_TYPES:
        filename = os.path.basename(file_path)
        if ext == ".pdf":
            text = extract_text_from_pdf(file_path)
        elif ext == ".doc" or ext == ".docx":
            text = extract_text_from_doc(file_path)
        doc = preprocess(text)
        persons, organizations = extractNER(doc)
        emails = extract_email_addresses(doc)
        phones = extract_phones(doc)
        educations = extractEducation(doc)
        links = set(extractLinks(doc))
        skills = set(extractSkills(text))
        out = {
            "filename": filename,
            "data": {
                "persons": persons,
                "organizations": organizations,
                "emails": list(emails),
                "phones": list(phones),
                "educations": list(educations),
                "links": list(links),
                "skills": list(skills)
            }
        }

    return json.dumps(out)



  # Usage
file_path = "/content/marketing-assistant-resume-example.pdf"
out = json.loads(resume_parser(file_path))
out


# Usage
file_path = "/content/sample-resume.docx"
out = json.loads(resume_parser(file_path))
out






def compute_cosine_similarity(text1, text2):
  text1 = list(text1)
  text2 = list(text2)

  # Create a set of unique keywords from both users and resume
  all_text = set(text1 + text2)

  # Create vectors for each skill set
  vector1 = np.array([1 if t in text1 else 0 for t in all_text])
  vector2 = np.array([1 if t in text2 else 0 for t in all_text])

  # Reshape vectors into 2D arrays
  vector1 = vector1.reshape(1, -1)
  vector2 = vector2.reshape(1, -1)

  return cosine_similarity(vector1, vector2)[0, 0]





def compute_jaccard_index(text1, text2):
  text1 = set(text1)
  text2 = set(text2)

  intersection = text1.intersection(text2)
  union = text1.union(text2)
  similarity = len(intersection) / len(union)

  return similarity





def similarity_score(source, target):
  weight_cosine = 0.7
  weight_jaccard = 0.3
  weight_em = 0.7
  weight_fm = 0.3

  # exact match
  text1 = [sublist[0] for sublist in source if sublist[1] == 1]
  text2 = [sublist[0] for sublist in target if sublist[1] == 1]

  cosine_similarity_score_em = compute_cosine_similarity(text1, text2)
  jaccard_similarity_score_em = compute_jaccard_index(text1, text2)

  # Fuzzy match
  text1 = [sublist[0] for sublist in text1]
  text2 = [sublist[0] for sublist in text2]
  cosine_similarity_score_fm = compute_cosine_similarity(text1, text2)
  jaccard_similarity_score_fm = compute_jaccard_index(text1, text2)

  cosine_similarity_score = ((cosine_similarity_score_fm * weight_fm) + (cosine_similarity_score_em * weight_em))
  jaccard_similarity_score = ((jaccard_similarity_score_fm * weight_fm) + (jaccard_similarity_score_em * weight_em))

  combined_score = (cosine_similarity_score * weight_cosine) + (jaccard_similarity_score * weight_jaccard)
  return combined_score



  text = "We are seeking a talented and experienced Marketing Manager to join our team. As a Marketing Manager, you will be responsible for developing and implementing strategic marketing plans to drive brand awareness, enhance customer engagement, and generate leads. You will work closely with cross-functional teams to execute marketing campaigns and initiatives that align with our business objectives. Develop and execute comprehensive marketing strategies to promote our products/services and increase market share. Conduct market research to identify customer needs, market trends, and competitive landscape. Plan and implement digital marketing campaigns across various platforms, including social media, email marketing, content marketing, and search engine optimization (SEO). Monitor and analyze campaign performance, using data-driven insights to optimize marketing activities and achieve KPIs. Collaborate with the creative team to develop compelling marketing materials, including website content, blog posts, videos, and infographics. Build and maintain strong relationships with key stakeholders, such as media partners, industry influencers, and customers. Manage the marketing budget effectively, allocating resources to maximize ROI and achieve marketing goals. Stay updated on emerging marketing trends, technologies, and best practices to drive innovation and maintain a competitive edge. Lead and mentor a team of marketing professionals, providing guidance, support, and performance feedback. Bachelor's degree in Marketing, Business Administration, or a related field. MBA is a plus. Proven experience in marketing, with a focus on developing and implementing successful marketing strategies. Strong understanding of digital marketing channels, including social media, SEO, content marketing, and email marketing. Experience in analyzing marketing data and using metrics to drive decision-making and campaign optimization. Excellent written and verbal communication skills, with the ability to create engaging content and present ideas effectively. Strong project management skills, with the ability to manage multiple priorities and deliver projects on time. Demonstrated leadership abilities, with experience in managing and developing a team. Creative thinker with a passion for marketing and a drive to stay updated on industry trends and best practices. Results-oriented mindset, with a focus on achieving measurable marketing objectives."




  job_desc_skills = list(extractSkills(text))


  text1 = [(sublist[1], sublist[2]) for sublist in job_desc_skills]
text2 = [(sublist[1], sublist[2]) for sublist in out['data']['skills']]
similarity_score(text1, text2)



import datetime
def main(zip_file_path, job_desc):

  # Open the zip file
  with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
      # Extract all the contents to the specified directory
      zip_ref.extractall(UNZIP_TO_FLD)

  # Extract skills from job_desc
  job_desc_skills = list(extractSkills(job_desc))
  skills_source = [(sublist[1], sublist[2]) for sublist in job_desc_skills]

  data = []
  counter = 1
  # Loop through the files in the folder
  for filename in os.listdir(UNZIP_TO_FLD):
      # Construct the absolute file path
      file_path = os.path.join(UNZIP_TO_FLD, filename)

      # Check if the current item is a file
      if os.path.isfile(file_path):
          # Perform operations on the file
          print("Processing file:", file_path)
          # Extract data from resume
          out = json.loads(resume_parser(file_path))
          skills_target = [(sublist[1], sublist[2]) for sublist in out["data"]["skills"]]
          match_score = similarity_score(skills_source, skills_target)
          ele = {
              "id": counter,
              "match_score": match_score,
              "name": out["data"]["persons"],
              "emails": out["data"]["emails"],
              "phones": out["data"]["phones"],
              "educations": out["data"]["educations"],
              "links": out["data"]["links"],
              "skills": skills_target,
              "resume": file_path
          }
          data.append(ele)
          counter += 1

  jout = {
    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "zip_file": zip_file_path,
    "data": data
  }

  return json.dumps(jout)




zip_file_path = "/content/marketing_resumes.zip"

job_desc = "We are seeking a talented and experienced Marketing Manager to join our team. As a Marketing Manager, you will be responsible for developing and implementing strategic marketing plans to drive brand awareness, enhance customer engagement, and generate leads. You will work closely with cross-functional teams to execute marketing campaigns and initiatives that align with our business objectives. Develop and execute comprehensive marketing strategies to promote our products/services and increase market share. Conduct market research to identify customer needs, market trends, and competitive landscape. Plan and implement digital marketing campaigns across various platforms, including social media, email marketing, content marketing, and search engine optimization (SEO). Monitor and analyze campaign performance, using data-driven insights to optimize marketing activities and achieve KPIs. Collaborate with the creative team to develop compelling marketing materials, including website content, blog posts, videos, and infographics. Build and maintain strong relationships with key stakeholders, such as media partners, industry influencers, and customers. Manage the marketing budget effectively, allocating resources to maximize ROI and achieve marketing goals. Stay updated on emerging marketing trends, technologies, and best practices to drive innovation and maintain a competitive edge. Lead and mentor a team of marketing professionals, providing guidance, support, and performance feedback. Bachelor's degree in Marketing, Business Administration, or a related field. MBA is a plus. Proven experience in marketing, with a focus on developing and implementing successful marketing strategies. Strong understanding of digital marketing channels, including social media, SEO, content marketing, and email marketing. Experience in analyzing marketing data and using metrics to drive decision-making and campaign optimization. Excellent written and verbal communication skills, with the ability to create engaging content and present ideas effectively. Strong project management skills, with the ability to manage multiple priorities and deliver projects on time. Demonstrated leadership abilities, with experience in managing and developing a team. Creative thinker with a passion for marketing and a drive to stay updated on industry trends and best practices. Results-oriented mindset, with a focus on achieving measurable marketing objectives."



results = main(zip_file_path, job_desc)


results_v = json.loads(results)


import pandas
df = pd.DataFrame(results_v["data"])


# Sort the DataFrame by the 'match score' column in descending order
sorted_df = df.sort_values(by='match_score', ascending = False)


print(sorted_df)
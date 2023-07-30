from django.shortcuts import render
from django.http import JsonResponse
import os
import zipfile
import json
from .resume_parser import preprocess, extractSkills, similarity_score


# Import your intelligent resume parser functions from intelligent_resume_parser.ipynb
# Replace 'your_parser_function' with the actual function name from the notebook
from intelligent_resume_parser import your_parser_function

# Define the view for the homepage
def home(request):
    return render(request, 'resume_parser_app/home.html')

# Define the view for processing the resumes and job description
def process_resumes(request):
    if request.method == 'POST':
        job_description = request.POST['job_description']
        resume_zip = request.FILES['resume_zip']
        
        # Save the uploaded zip file to a temporary location
        with open('temp.zip', 'wb+') as destination:
            for chunk in resume_zip.chunks():
                destination.write(chunk)
        
        # Call the intelligent resume parser code here
        # Assuming your intelligent parser function takes the path to the zip file and job description as inputs
        # Replace 'your_parser_function' with the actual function name from the notebook
        parsed_result = your_parser_function('temp.zip', job_description)

        # Process the resumes and job description, then return the results as a JSON response
        return JsonResponse(parsed_result)

    return render(request, 'resume_parser_app/home.html')

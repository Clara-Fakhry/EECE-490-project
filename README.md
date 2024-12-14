# Career guidance system
Career Guidance System is an AI-powered application designed to assist fresh graduates in finding jobs that best suit their profiles. The application analyzes the user’s CV to extract features and skills, then presents career recommendations tailored to their qualifications. For each suggested career path, the system evaluates the candidate's strengths and weaknesses, providing insights to help them assess their chances of success and identify areas for improvement.  
To enhance the user experience, the application invites users to rate the proposed job suggestions and provide feedback. This feedback is used to refine career recommendations. Additionally, the program creates specific goals for each career path, broken into manageable steps. Users can track their progress toward these goals, and the app offers resources to support them in completing the remaining steps.




# What's new
-	26 October 2024: version 1.1.1 | cv analysis - CV extraction with initial recommendation 

-	4 November 2024:  version 1.1.2 | advanced job analysis -  Added role analysis and skill match

-	12 November 2024 : version 1.1.3 | feedback and goal tracking - Added user feedback and SMART goal generation

-	27 November 2024: version 1.2.1 | User interface - Integrated backend program into UI

-	7 December 2024: version 1.2.2 | system english -  Fixed bugs and added loop for feedback 

-	10 December 2024: version 1.2.3 | final app - Added multi-language support
  
The first few versions (1.1) where only developed as a backend and are not released. Versions 1.2 contain both the backend and the frontend UI. Note that the latest version (1.2.3 | finalapp) contains multi-language support but it is slower and still contain unresolved issues or bugs. If you face problems, use version 1.2.2 | system english which is stable and fully tested.






# Usage

### Using Docker
**Step 1:** provide your openai API key in the code  
```python
openai.api_key= "insert key here"
```

**Step 2:** build and run the docker image using the following command   
- For latest stable version (systemenglish.py):  
```
docker build -f Dockerfile.systemenglish -t systemenglish-image .
docker run -p 8505:8505 systemenglish-image 
```
-	For latest release with multilanguage support (finalapp.py):  
```
docker build -f Dockerfile.finalapp -t finalapp-image .
docker run -p 8506:8506 finalapp-image
```

**Step 3:** a link will appear on the terminal. Click on it to open the UI in your browser. 

### Without Docker
**Step 1:** make sure to install all the needed libraries (check requirements.txt)

**Step 2:** provide your openai API key in the code  
```python
openai.api_key= "insert key here"
```

**Step 3:** Run the code and write the following command in the terminal  
-	For latest stable version (systemenglish):  
```python
streamlit run systemenglish.py
```
-	For latest release with multilanguage support (finalapp):  
```python
streamlit run finalapp.py
```



# Code flow
![image](https://github.com/user-attachments/assets/271f2432-b7a3-4d7f-8a1a-eaf75cd1d5e3)

# Contributors 

# Acknowledgement



import fitz  # PyMuPDF
import openai
import uuid
import os
import io
import fitz
import mimetypes
import streamlit as st
import tempfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from fpdf import FPDF
import json

#backend
def save_goal_progress_to_file(filename="goal_progress.json"):
    try:
        # Save the global goal_completion_tracker to a JSON file
        with open(filename, 'w') as f:
            json.dump(goal_completion_tracker, f, indent=4)
        print(f"Goal progress successfully saved to {filename}")
    except Exception as e:
        print(f"Error saving goal progress: {e}")
def load_goal_progress_from_file(filename="goal_progress.json"):
    try:
        # Load the global goal_completion_tracker from the JSON file
        with open(filename, 'r') as f:
            global goal_completion_tracker
            goal_completion_tracker = json.load(f)
        print(f"Goal progress successfully loaded from {filename}")
    except FileNotFoundError:
        print(f"No saved progress found. Starting fresh.")
    except Exception as e:
        print(f"Error loading goal progress: {e}")

import json
import os

# Function to save data to JSON file
def save_to_json(data, filename="user_data.json"):
    try:
        with open(filename, "w") as json_file:
            json.dump(data, json_file)
        st.success("Progress saved successfully!")
    except Exception as e:
        st.error(f"Error saving data: {e}")

# Function to load data from JSON file
def load_from_json(filename="user_data.json"):
    if os.path.exists(filename):
        try:
            with open(filename, "r") as json_file:
                data = json.load(json_file)
            return data
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None
    else:
        return None


# Set your API key securely (e.g., using environment variables)
openai.api_key = "api_key"  # Or use getpass to input securely

def get_chatgpt_response(prompt, model="gpt-4"):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message['content']

# Helper function to validate if the uploaded file is a PDF
def is_valid_pdf(uploaded_file):
    mime_type, _ = mimetypes.guess_type(uploaded_file.name)
    return mime_type == "application/pdf"

import tempfile
import shutil

def extract_text_from_pdf(uploaded_file):
    # Create a temporary file to save the uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_file_path = tmp_file.name

    try:
        with fitz.open(temp_file_path) as doc:
            # Extract text from the PDF document
            text = ""
            for page in doc:
                text += page.get_text()

        # Clean up the temporary file
        os.remove(temp_file_path)
        return text
    except Exception as e:
        raise Exception(f"Error processing PDF: {e}")

       





def get_cv_features(cv_text):
    prompt = f"""
    Extract the following details from the CV text below and structure them in a clear format:

    1. Contact Information (e.g., name, phone number, email, address).
    2. Summary or Objective (if available).
    3. Education (degrees, institutions, graduation dates, GPA or academic achievements).
    4. Work Experience (company names, job titles, responsibilities, achievements, dates).
    5. Skills (both technical and soft skills).
    6. Certifications or Training (certificate names, issuing organizations, dates).
    7. Achievements and Awards (e.g., recognitions, honors, major accomplishments).
    8. Extracurricular Activities (e.g., clubs, volunteer work, sports).
    9. Languages (languages spoken and proficiency levels).
    10. Projects (if any relevant projects are mentioned).
    11. References (if any references are included).

    CV Text:
    {cv_text}
    """
    
    extracted_info = get_chatgpt_response(prompt)
    return extracted_info

def generate_potential_roles(cv_features):
    prompt = f"""
    Given the following CV features, suggest a list of potential job roles that may suit the candidate.
    Consider roles that align with the candidate's skills, experience, and education, and provide a brief
    explanation of why each role may be a good fit.

    CV Features:
    {cv_features}
    """
    potential_roles = get_chatgpt_response(prompt)
    return potential_roles

def analyze_roles_with_cv(cv_features, potential_roles):
    prompt = f"""
    Based on the following potential job roles and CV features, identify the candidate's strengths and gaps for each role.
    For each role, outline which skills or experiences are strong matches and which may require development. Minimize your wordings as much as possible while keeping the information concise.

    Potential Job Roles:
    {potential_roles}

    CV Features:
    {cv_features}
    """
    role_analysis = get_chatgpt_response(prompt)
    return role_analysis

def get_career_path_descriptions(potential_roles):
    prompt = f"""
    For each of the following job roles, provide a detailed description including:
    - Key skill requirements
    - Current industry trends
    - Potential growth areas
    Minimize your wordings as much as possible while keeping the information concise.

    Job Roles:
    {potential_roles}
    """
    career_path_descriptions = get_chatgpt_response(prompt)
    return career_path_descriptions


def calculate_skill_match(cv_features, refined_recommendations):
    # Construct the prompt for GPT
    prompt = f"""
    For each of the refined job roles provided, calculate a skill match percentage based on the candidate's skills from their CV.
    Match each skill mentioned in the CV to the skills required for the job role, and return a percentage match score.
    Additionally, provide brief reasoning for the score based on matching and missing skills, but when you give me the refined recommendations and the refined skill match percentages, dont give me the skill match percentages of the old recommendations.
    Minimize your wordings as much as possible while keeping the information concise.

    CV Skills:
    {cv_features}

    Refined Job Roles:
    {refined_recommendations}
    """

    # Call the GPT model to process the prompt
    skill_match_data = get_chatgpt_response(prompt)
    
    # Parsing the response to extract required components
    refined_recommendations = skill_match_data.get("refined_recommendations", [])
    skill_matches = skill_match_data.get("skill_matches", {})
    reasonings = skill_match_data.get("reasonings", {})

    return refined_recommendations, skill_matches, reasonings


# Step 2: Collect user feedback and refine recommendations with skill match percentages
def collect_user_feedback_and_refine_recommendations(recommendations, potential_roles, role_analysis, cv_features):
    while True:
        try:
            rating = int(input("On a scale of 1 to 10, how would you rate the relevance of these career suggestions? "))
            if 1 <= rating <= 10:
                break
            else:
                print("Please enter a rating between 1 and 10.")
        except ValueError:
            print("Please enter a valid integer between 1 and 10.")

    feedback = input("Provide any comments on the career suggestions (mention likes/dislikes): ")

    # Refine recommendations based on user feedback
    prompt = f"""
    Based on the initial recommendations and the user's feedback, provide refined career suggestions.
    Adjust recommendations based on the candidate's preferences and skill match percentage.
    Keep the information concise.

    Initial Recommendations:
    {recommendations}

    User Feedback:
    Rating: {rating}/10
    Comments: {feedback}

    CV Features:
    {cv_features}

    Previous Potential Job Roles:
    {potential_roles}

    Previous Role Analysis:
    {role_analysis}
    """
    refined_recommendations = get_chatgpt_response(prompt)
    refined_skill_match_data = calculate_skill_match(cv_features, potential_roles)

    return {
        "rating": rating,
        "feedback": feedback,
        "refined_recommendations": refined_recommendations,
        "refined_skill_match_data": refined_skill_match_data,
    }


def career_recommendation_engine_with_feedback_and_skill_match(pdf_path):
    # Step 1: Extract and analyze CV information
    cv_text = extract_text_from_pdf(pdf_path)
    cv_features = get_cv_features(cv_text)
    potential_roles = generate_potential_roles(cv_features)
    role_analysis = analyze_roles_with_cv(cv_features, potential_roles)
    career_path_descriptions = get_career_path_descriptions(potential_roles)

    # Step 2: Calculate initial skill match percentages
    initial_skill_match_data = calculate_skill_match(cv_features, potential_roles)
    initial_recommendations = f"""
    Career Recommendations for the Candidate:
    Extracted CV Features: {cv_features}
    Recommended Job Roles: {potential_roles}
    Role Analysis: {role_analysis}
    Career Path Descriptions: {career_path_descriptions}
    Initial Skill Match Percentages: {initial_skill_match_data}
    """
    print(initial_recommendations)

    # Step 3: Collect feedback and refine recommendations
    user_feedback = collect_user_feedback_and_refine_recommendations(
        recommendations=initial_recommendations,
        potential_roles=potential_roles,
        role_analysis=role_analysis,
        cv_features=cv_features,
    )

    return user_feedback["refined_recommendations"], user_feedback["refined_skill_match_data"]

 

# Function to identify skill gaps and recommend resources based on refined career suggestions
def identify_skill_gaps_and_recommend_resources(cv_features, refined_recommendations):
    # Define prompt to analyze gaps and recommend resources
    prompt = f"""
    Based on the user's current skill set in their CV and the refined career recommendations, identify any skill gaps for each recommended career path.
    For each skill gap, provide actionable recommendations including courses, certifications, project ideas, and other resources to help the user develop these skills.

    The recommendations should include:
    - Specific learning platforms (like Coursera, LinkedIn Learning, Udemy, edX)
    - Suggested project ideas to apply the skills practically
    - Mentorship or networking suggestions if applicable
    Minimize your wordings as much as possible while keeping the information concise.

    User's Current Skills:
    {cv_features}

    Refined Career Recommendations and Required Skills:
    {refined_recommendations}
    """

    # Call the OpenAI API to get skill gap analysis and recommendations


    # Extract the content of the response for skill gap recommendations
    skill_gap_recommendations = get_chatgpt_response(prompt)

    return skill_gap_recommendations

# Main function to execute skill gap analysis and resource recommendation
def skill_gap_analysis_with_recommendations(pdf_path):
    # Extract and analyze CV information
    cv_text = extract_text_from_pdf(pdf_path)
    cv_features = get_cv_features(cv_text)

    # Generate initial recommendations and refine based on feedback
    refined_recommendations, refined_skill_match_data = career_recommendation_engine_with_feedback_and_skill_match(pdf_path)


    # Perform skill gap analysis on refined recommendations
    skill_gap_recommendations = identify_skill_gaps_and_recommend_resources(cv_features, refined_recommendations)

    # Display the skill gap recommendations
    print("\nSkill Gap Analysis and Recommendations:\n")
    print(skill_gap_recommendations)

def get_motivational_quote():
    quote_prompt = "Give me an uplifting and motivational quote to inspire someone who's working on their career goals."
    quote= get_chatgpt_response(quote_prompt)
    return quote 


def calculate_goal_progress(steps, completed_steps):
    # Edge case 1: If there are no steps, return 0% progress
    if not steps:
        print("No steps defined for this goal.")
        return 0.0
    
    # Edge case 2: If there are no completed steps, return 0% progress
    if not completed_steps:
        return 0.0
    
    # Calculate progress based on the number of completed steps
    progress = (len(completed_steps) / len(steps)) * 100
    
    return progress



# Function to generate SMART goals and steps for a career recommendation
def generate_smart_goals(refined_recommendation):
    prompt = f"""
    For the career recommendation '{refined_recommendation}', create 2–3 SMART goals.
    Each goal should include:
    - A clear and specific description (Specific, Measurable, Achievable, Relevant, Time-bound).
    - 3–5 sequential steps to achieve the goal, formatted as a numbered list.
    Ensure the output is concise, well-structured, and easy to read.
    When you provide the steps to achieve, I want it to have this header:'Steps to Achieve:'
    """
    
    smart_goals = get_chatgpt_response(prompt)
    return smart_goals
# Helper: Extract unique career titles from refined recommendations
def extract_career_titles(refined_recommendations):
    titles = set()  # Use a set to avoid duplicates
    for line in refined_recommendations.split("\n"):
        # Identify lines that start with numbers followed by a period (e.g., '1. ')
        if line.strip().startswith(tuple(str(i) + ". " for i in range(1, 10))):
            # Extract the title part (after the number and period)
            title = line.split(". ", 1)[1].strip()
            # Remove percentages or redundant text
            title = title.split(":")[0].strip()  # Remove text after colons, like percentages
            titles.add(title)
    return list(titles)  # Convert back to a list for ordered display




# Global dictionary to store completed steps for each goal
goal_completion_tracker = {}
# Main function for career goal tracking
def career_goal_tracking(pdf_path, career_titles=None, smart_goals_dict=None):
    global goal_completion_tracker  # Use the global tracker to retain the completed steps across interactions

    # If no career_titles and smart_goals_dict are provided, generate them
    if career_titles is None or smart_goals_dict is None:
        # Step 1: Extract CV information and generate refined recommendations
        cv_text = extract_text_from_pdf(pdf_path)
        cv_features = get_cv_features(cv_text)


        refined_recommendations, _ = career_recommendation_engine_with_feedback_and_skill_match(pdf_path)
        
        # Extract career titles
        career_titles = extract_career_titles(refined_recommendations)
        

        # Step 2: Generate SMART goals and steps for all career recommendations
        smart_goals_dict = {}
        print("\n--- Generating SMART Goals for Career Recommendations ---")
        for title in career_titles:
            print(f"Generating goals for: {title}")
            smart_goals = generate_smart_goals(title)
            smart_goals_dict[title] = smart_goals
            print(f"\nSMART Goals for {title}:")
            print(smart_goals)
            print("--------------------------------------")
        print("\nSMART goals and steps are ready!")

    # Step 3: Start user interaction
    # Display career titles as a clean list
    print("\n--- Refined Career Recommendations ---")
    for i, title in enumerate(career_titles, 1):
        print(f"{i}. {title}")
    print("--------------------------------------")

    # User selects a career recommendation
    selected_index = int(input("\nSelect a career recommendation (enter the number): ")) - 1
    selected_recommendation = career_titles[selected_index]
    print(f"\nSelected Career Recommendation: {selected_recommendation}")

    # Display SMART goals for the selected recommendation
    print("\n--- SMART Goals for Selected Recommendation ---")
    smart_goals_text = smart_goals_dict[selected_recommendation]

    # Split SMART goals into individual goals using a consistent delimiter
    goals = smart_goals_text.split("\n### Goal")  # Split by "### Goal" to isolate each goal

    # Prepend "### Goal" to each split goal for proper formatting
    goals = [f"### Goal{goal}" for goal in goals if goal.strip()]  # Ensure formatting remains intact

    # Check if goals are properly extracted
    if not goals or all(goal.strip() == "" for goal in goals):
        print("No SMART goals were generated for this recommendation. Please try again.")
        return  # Exit gracefully if no goals are found

    # Display all extracted goals with proper enumeration
    for i, goal in enumerate(goals, 1):
        # Show only the title of each goal
        title = goal.strip().splitlines()[0] if goal.strip() else "Untitled Goal"
        print(f"{i}. {title}")
    print("-----------------------------------------------")

    # User selects a goal with validation
    while True:
        try:
            goal_index = int(input("\nSelect a goal (enter the number): ")) - 1
            if goal_index < 0 or goal_index >= len(goals):
                raise ValueError("Invalid selection. Please enter a valid number.")
            break
        except ValueError as e:
            print(f"Invalid input: {e}. Please try again.")

    # Isolate the selected goal
    selected_goal = goals[goal_index].strip()

    # Display the selected goal
    print(f"\nSelected Goal:\n{selected_goal}")

    # Extract steps for the selected goal
    steps = []  # Initialize steps as an empty list

    # Attempt to locate a steps section by searching for common headers
    possible_headers = ["**Steps to Achieve:**", "**Steps to Achieve the Goal:**", "**Steps:**", "**Steps to Achieve this Goal:**","Steps to Achieve:"]
    steps_start = -1  # Initialize with -1 to indicate no header found

    # Find the starting position of the steps section
    for header in possible_headers:
        steps_start = selected_goal.find(header)
        if steps_start != -1:  # If a header is found, stop the search
            break

    # If a steps section header is found, extract and parse the steps
    if steps_start != -1:
        # Extract the portion of the text after the header
        steps_section = selected_goal[steps_start + len(header):].strip()

        # Parse steps: Look for numbered lines and stop at unrelated sections
        for line in steps_section.split("\n"):
            line = line.strip()
            if line and line[0].isdigit():  # Identify numbered steps
                steps.append(line)
            elif line.startswith("**"):  # Stop parsing at a new section header
                break
    else:
        # Handle case where no steps header is found
        print("\nNo steps section header detected in the goal. Please verify the goal format.")

    # Display steps for the selected goal
    if not steps:
        print("\nNo steps available for this goal. Please check the goal setup.")
    else:
        print("\n--- Steps for the Selected Goal ---")
        for i, step in enumerate(steps, 1):  # Display steps with proper numbering
            print(f"- {step}")
        print("-----------------------------------")

        # Retrieve previously completed steps from the tracker
        if selected_goal not in goal_completion_tracker:
            goal_completion_tracker[selected_goal] = []

        completed_steps = goal_completion_tracker[selected_goal]

        # User marks completed steps with validation
        while True:
            try:
                # Prompt user for completed steps
                completed_input = input("\nEnter the numbers of completed steps (comma-separated): ").strip()

                # Ensure input is not empty
                if not completed_input:
                    raise ValueError("You must enter at least one step number.")

                # Parse input and convert to zero-based indices
                completed_indices = [int(idx.strip()) - 1 for idx in completed_input.split(",")]

                # Validate indices against the steps list
                if any(idx < 0 or idx >= len(steps) for idx in completed_indices):
                    raise ValueError("Some indices are out of range. Please ensure your input matches the listed steps.")

                # Check for duplicate steps (i.e., the user should not re-enter steps they've already completed)
                already_completed = [steps[idx] for idx in completed_indices if steps[idx] in completed_steps]
                if already_completed:
                    raise ValueError(f"You've already marked these steps as completed: {', '.join(already_completed)}. Please only enter new steps.")

                # If all validations pass, break the loop
                break
            except ValueError as e:
                print(f"Invalid input: {e}. Please try again.")

        # Retrieve completed steps and update the tracker
        completed_steps.extend([steps[idx] for idx in completed_indices])

        # Store the updated list in the global tracker
        goal_completion_tracker[selected_goal] = completed_steps

        # Display completed steps
        print("\nCompleted Steps:")
        for step in completed_steps:
            print(f"- {step}")

    # Step 5: Calculate progress
    progress = calculate_goal_progress(steps, completed_steps)
    print(f"\nGoal Completion Progress: {progress:.2f}%")

    # Provide motivational advice
    if progress < 100:
        incomplete_steps = [step for step in steps if step not in completed_steps]
        print("\nMotivational Feedback: Keep pushing! You’re making progress.")
        print("Here are additional resources to help you complete the remaining steps:")
        resource_prompt = f"""
        Provide motivational advice and resources for the following incomplete steps:
        {incomplete_steps}
        """

        motivational_feedback = get_chatgpt_response(resource_prompt)
        print(motivational_feedback)

    # Ask user if they'd like to explore another career path
    while True:
        continue_choice = input("\nWould you like to explore another career path? (yes/no): ").strip().lower()
        if continue_choice == "yes":
            career_goal_tracking(pdf_path, career_titles, smart_goals_dict)
            break
        elif continue_choice == "no":
            quote=get_motivational_quote()
            print(quote)
            print("\nGood luck on your journey!")
            break
        else:
            print("Invalid input. Please enter 'yes' or 'no'.")


def collect_user_feedback_and_refine_recommendations_ui(potential_roles, role_analysis, cv_features):
    # Collect feedback from user
    st.subheader("Feedback on Recommendations")
    
    # Rating slider and text area for feedback
    rating = st.slider("Rate these recommendations (1-10):", 1, 10, key="rating_slider")
    feedback = st.text_area("Your feedback on the recommendations:", key="feedback_area")

    # Check if both rating and feedback are provided before refining
    if st.button("Submit Feedback"):
        if rating and feedback.strip():
            st.success("Processing your feedback...")

            # Creating the prompt for refinement based on user feedback
            prompt = f"""
            Refine career suggestions based on:
            - Rating: {rating}/10
            - Feedback: {feedback.strip()}
            - CV Features: {cv_features}
            - Potential Roles: {potential_roles}
            - Role Analysis: {role_analysis}
            """
            
            try:
                # Get refined recommendations using an external function (e.g., GPT-3/ChatGPT)
                refined_recommendations = get_chatgpt_response(prompt)
                
                # Calculate skill match for refined recommendations
                refined_recommendations, skill_matches, reasonings = calculate_skill_match(cv_features, refined_recommendations)
                
                # Display refined recommendations and skill match
                #st.subheader("Refined Recommendations")
                #st.write(refined_recommendations)
                #st.subheader("Skill Match Percentages")
                #st.write(refined_skill_match)
                
                # Return the refined data for later use
                return refined_recommendations, skill_matches, reasonings
            except Exception as e:
                st.error(f"Error during refinement: {e}")
        else:
            st.error("Please provide both a rating and feedback.")
    
    # If no feedback submitted yet, return None to indicate no refinement
    return None, None

def career_recommendation_engine_with_feedback_and_skill_match_ui(uploaded_file):
    cv_text = extract_text_from_pdf(uploaded_file)
    cv_features = get_cv_features(cv_text)
    potential_roles = generate_potential_roles(cv_features)
    role_analysis = analyze_roles_with_cv(cv_features, potential_roles)
    career_path_descriptions = get_career_path_descriptions(potential_roles)

    st.subheader("Initial Career Recommendations")
    st.text(f"""
    CV Features:
    {cv_features}
    Potential Roles:
    {potential_roles}
    Role Analysis:
    {role_analysis}
    Career Paths:
    {career_path_descriptions}
    """)

    collect_user_feedback_and_refine_recommendations_ui(
        recommendations=potential_roles,
        potential_roles=potential_roles,
        role_analysis=role_analysis,
        cv_features=cv_features
    )

# Global dictionary to store completed steps for each goal
goal_completion_tracker_ui = {}


# Function to generate the PDF
def generate_pdf(content):
    pdf = fpdf.FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Arial", size=12)
    
    # Add content to the PDF
    for line in content:
        pdf.cell(200, 10, txt=line, ln=True, align="L")
    
    # Save the PDF to a file
    pdf_output = "/tmp/career_recommendations.pdf"
    pdf.output(pdf_output)


def generate_full_report():
    # Initialize PDF object
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Set Title for the document
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Career Recommendations and SMART Goals Report", ln=True, align="C")
    pdf.ln(10)

    # Section 1: Welcome Message & Motivational Quote
    pdf.set_font("Arial", 'I', 12)
    pdf.multi_cell(0, 10, "Welcome to the Career Recommendation System!\n")
    pdf.multi_cell(0, 10, "Your journey towards a better career starts here!")
    pdf.ln(5)
    
    motivational_quote = "Success is the sum of small efforts, repeated day in and day out."
    pdf.multi_cell(0, 10, f"Motivational Quote: {motivational_quote}")
    pdf.ln(10)

    # Section 2: CV Information (if uploaded)
    if "cv_text" in st.session_state:
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10, "Uploaded CV Text:", ln=True)
        pdf.set_font("Arial", '', 12)
        pdf.multi_cell(0, 10, st.session_state["cv_text"])
        pdf.ln(10)

        # CV Features
        if "cv_features" in st.session_state:
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(200, 10, "Extracted CV Features:", ln=True)
            pdf.set_font("Arial", '', 12)
            for feature, value in st.session_state["cv_features"].items():
                pdf.multi_cell(0, 10, f"{feature}: {value}")
            pdf.ln(10)

    # Section 3: Career Recommendations
    if "potential_roles" in st.session_state:
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10, "Career Recommendations:", ln=True)
        pdf.set_font("Arial", '', 12)
        for role in st.session_state["potential_roles"]:
            pdf.multi_cell(0, 10, f"- {role}")
        pdf.ln(10)

    # Section 4: Feedback and Refinement
    if "refined_recommendations" in st.session_state:
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10, "Refined Career Recommendations:", ln=True)
        pdf.set_font("Arial", '', 12)
        for role in st.session_state["refined_recommendations"]:
            pdf.multi_cell(0, 10, f"- {role}")
        pdf.ln(10)

        # Skill Match Data (if any)
        if "refined_matches" in st.session_state:
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(200, 10, "Skill Match Percentages:", ln=True)
            pdf.set_font("Arial", '', 12)
            for role, match in zip(st.session_state["refined_recommendations"], st.session_state["refined_matches"]):
                pdf.multi_cell(0, 10, f"{role}: {match}% match")
            pdf.ln(10)

    # Section 5: SMART Goals
    if "smart_goals" in st.session_state:
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10, "SMART Goals:", ln=True)
        pdf.set_font("Arial", '', 12)
        for role, goals in st.session_state["smart_goals"].items():
            pdf.multi_cell(0, 10, f"SMART Goals for {role}:")
            pdf.multi_cell(0, 10, goals)
            pdf.ln(10)

    # Save the PDF to a file
    pdf_file_path = "/path/to/your/pdf_file.pdf"  # Modify the path as needed
    pdf.output(pdf_file_path)

    return pdf_file_path



# Backend function to generate an image based on a prompt
def generate_dynamic_image(prompt):
    try:
        # Request to DALL·E for image generation
        response = openai.Image.create(
            prompt=prompt,  # The user-specific prompt for image generation
            n=1,  # Number of images to generate
            size="1024x1024"  # Image size (can adjust as needed)
        )
        # Extract the URL of the generated image
        image_url = response['data'][0]['url']
        return image_url
    except Exception as e:
        st.error(f"Error generating image: {e}")
        return None

def extract_steps(selected_goal):
    """
    Extract actionable steps from a SMART goal description.

    Args:
    selected_goal (str): A string containing the description of the SMART goal.

    Returns:
    list: A list of steps to be completed for the goal.
    """
    # Assuming steps are listed in a numbered format within the goal text, e.g.:
    # "Step 1: Do X"
    # "Step 2: Complete Y"
    # Extracting steps using a simple pattern
    steps = []
    
    # Split the goal description into lines, assuming each line might represent a step
    lines = selected_goal.split("\n")
    
    # Filter lines that represent steps (e.g., lines that contain "Step" or are part of the goal breakdown)
    for line in lines:
        # Here, we're looking for lines that start with "Step" or have any actionable task
        if line.strip().lower().startswith("step") or line.strip().lower().startswith("action"):
            steps.append(line.strip())
    
    # Return the list of extracted steps
    return steps

    
# Visual Helper Functions
def plot_bubble_chart(data, x_col, y_col, size_col, title="Bubble Chart"):
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=x_col, y=y_col, size=size_col, data=data, legend=False, sizes=(20, 200), alpha=0.6)
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    st.pyplot()

def plot_progress_chart(progress):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(["Career Progress"], [progress], color='royalblue')
    ax.barh(["Remaining Progress"], [100 - progress], color='lightgray')
    ax.set_xlim(0, 100)
    ax.set_xlabel("Progress (%)")
    ax.set_title("Goal Progress")
    st.pyplot(fig)

import streamlit as st
import plotly.graph_objects as go





def app():
    global goal_completion_tracker
    # Streamlit App
    st.set_page_config(page_title="Career Recommendation System", layout="wide")

    st.markdown("""
    <style>
    /* Main content area background */
    .stApp {
        background-color: #d4edda !important; /* Light success green background */
    }

    /* Global text color */
    h1, h2, h3, h4, h5, h6, p, div, span, li {
        color: #155724 !important; /* Dark success green text */
        font-family: 'Arial', sans-serif;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #76c893 !important; /* Light green for sidebar */
        color: white !important;
    }

    /* Sidebar links */
    [data-testid="stSidebar"] a {
        color: white !important;
        text-decoration: none !important;
        font-size: 1rem !important;
    }

    /* Buttons */
    .stButton button, .stFileUploader button {
        background-color: #76c893 !important; /* Green buttons */
        color: white !important;
        border-radius: 10px;
        padding: 10px 20px;
        border: none;
    }
    .stButton button:hover, .stFileUploader button:hover {
        background-color: #34a0a4 !important; /* Teal on hover */
    }
    </style>
""", unsafe_allow_html=True)







    # Sidebar Navigation
    st.sidebar.title("Career Navigator")
    navigation = st.sidebar.radio(
        "Choose a section:",
        ["Home", "Upload CV", "View Recommendations", "Feedback & Refinement", "Generate and Track SMART Goals", "Motivational Quotes"]
    )

    # Home Page with Motivational Quote and Image
    if navigation == "Home":
        st.title("Welcome to the Career Recommendation System!")
        st.subheader("Your journey towards a better career starts here!")
        
        motivational_quote = "Success is the sum of small efforts, repeated day in and day out."
        st.write("Success is the sum of small efforts, repeated day in and day out.")
        
        
        st.image("https://i.imgur.com/FZSPFtJ.jpeg", caption="Your Future Awaits", use_container_width=True, width=50)
    
   

    # Upload CV Section
    elif navigation == "Upload CV":
        st.title("Upload Your CV")
        uploaded_file = st.file_uploader("Upload your CV (PDF format)", type=["pdf"])
        if uploaded_file:
            with st.spinner("Extracting CV text..."):
                cv_text = extract_text_from_pdf(uploaded_file)
            st.success("CV text extracted successfully!")
            st.text_area("Extracted CV Text", cv_text, height=400)
            st.session_state["cv_text"] = cv_text  # Save to session_state
            st.session_state["uploaded_file"]=uploaded_file

    # View Recommendations Section
    elif navigation == "View Recommendations":
        st.title("Career Recommendations")
        
        # Check if the CV text exists in the session state
        if "cv_text" in st.session_state:
            if "cv_features" not in st.session_state:
                with st.spinner("Analyzing CV features..."):
                    cv_features = get_cv_features(st.session_state["cv_text"])
                st.session_state["cv_features"] = cv_features  # Save to session_state
            if "potential_roles" not in st.session_state:
                with st.spinner("Generating potential job roles..."):
                    potential_roles = generate_potential_roles(cv_features)
                st.session_state["potential_roles"] = potential_roles  # Save to session_state

            if "role_analysis" not in st.session_state:
                with st.spinner("Analyzing roles with CV..."):
                    role_analysis = analyze_roles_with_cv(cv_features, potential_roles)
                st.session_state["role_analysis"] = role_analysis  # Save to session_state

            st.subheader("Potential Job Roles")
            st.write(st.session_state["potential_roles"])
        else:
            st.warning("Please upload your CV first!")



        
    #Feedback and recommendation section
    elif navigation == "Feedback & Refinement":
        st.title("Feedback & Refine Recommendations")
        
        # Ensure necessary data exists in session_state
        if "cv_text" in st.session_state:
            if "cv_features" not in st.session_state or "potential_roles" not in st.session_state:
                st.warning("Please generate recommendations first in the 'View Recommendations' section.")
            else:
                # Access necessary session data for refinement
                potential_roles = st.session_state["potential_roles"]
                role_analysis = st.session_state["role_analysis"]
                cv_features = st.session_state["cv_features"]
                
                # Call the feedback collection and refinement function
                refined_recommendations, refined_matches, reasonings = collect_user_feedback_and_refine_recommendations_ui(
                    potential_roles, role_analysis, cv_features
                )
                career_titles = extract_career_titles(refined_recommendations)

                # Display refined recommendations if the user submitted feedback
                if refined_recommendations and refined_matches:
                    st.success("Recommendations refined successfully!")

                    # Display refined recommendations
                    st.subheader("✨ Refined Career Recommendations")
                    for idx, role in enumerate(career_titles, start=1):
                        st.markdown(f"**{idx}. {role}**")

                    # Display skill match data as a styled dataframe
                    st.subheader("📊 Skill Match Percentages")
                    skill_match_df = pd.DataFrame({
                        "Job Role": career_titles,
                        "Skill Match (%)": refined_matches,
                    })
                    st.table(skill_match_df)

                    # Display reasoning for matches
                    st.subheader("🔍 Reasoning for Matches")
                    for idx, role in enumerate(career_titles, start=1):
                        st.markdown(f"**{idx}. {role}:** {reasonings[idx - 1]}")

                    # Save refined data to session state for later use
                    st.session_state["refined_recommendations"] = refined_recommendations
                else:
                    # If no feedback is given, prompt user for it
                    st.warning("Please provide feedback to refine the recommendations.")
        else:
            # If no CV is uploaded yet, prompt the user to upload it
            st.warning("Please upload your CV first!")





#  Generate and Track Smart Goals
    elif navigation == "Generate and Track SMART Goals":
        st.title("Generate and Track SMART Goals")

        # Validate that refined recommendations exist
        if "refined_recommendations" not in st.session_state or not st.session_state["refined_recommendations"]:
            st.warning("Please refine recommendations first in the 'Feedback & Refinement' section.")
        else:
            # Ensure SMART goals are generated only once
            if "smart_goals" not in st.session_state:
                refined_recommendations = st.session_state["refined_recommendations"]
                career_titles = extract_career_titles(refined_recommendations)
                st.session_state["career_titles"] = career_titles
                st.session_state["smart_goals"] = {}
                for title in career_titles:
                    with st.spinner(f"Generating SMART goals for {title}..."):
                        st.session_state["smart_goals"][title] = generate_smart_goals(title)

            # Dropdown to choose a career recommendation
            selected_role = st.selectbox(
                "Select a career recommendation to track goals:",
                ["Select a career recommendation"] + list(st.session_state["smart_goals"].keys())
            )

            # Store the user's selected role in session state
            if selected_role != "Select a career recommendation":
                st.session_state["selected_role"] = selected_role

            # Display SMART goals for the selected role
            if "selected_role" in st.session_state:
                selected_role = st.session_state["selected_role"]
                smart_goals_text = st.session_state["smart_goals"].get(selected_role, "")
                st.subheader(f"SMART Goals for {selected_role}")
                st.write(smart_goals_text)

                # Extract goal titles and display progress tracking
                goal_entries = [goal.strip() for goal in smart_goals_text.split("\n") if goal.strip()]
                full_goals = {}
                for entry in goal_entries:
                    if entry.startswith("Goal"):
                        title = entry.split(":", 1)[0]
                        description = entry.split(":", 1)[-1].strip().split(".")[0]
                        full_goals[f"{title}: {description}"] = entry

                goal_titles = list(full_goals.keys())
                selected_goal_title = st.selectbox("Select a goal to track:", ["Select a goal"] + goal_titles)

                if selected_goal_title == "Select a goal":
                    st.info("Please select a goal to proceed.")
                else:
                    # Track progress for the selected goal
                    st.subheader(f"Tracking Progress for: {selected_goal_title}")
                    selected_goal = full_goals[selected_goal_title]
                    steps = extract_steps(selected_goal)  # Assuming `extract_steps` is a helper function
                    completed_steps = st.session_state.get(selected_goal_title, [])

                    for i, step in enumerate(steps, 1):
                        st.write(f"{i}. {step}")

                    # Input completed steps
                    completed_input = st.text_input("Enter completed step numbers (comma-separated):")
                    if completed_input:
                        try:
                            completed_indices = [
                                int(idx.strip()) - 1 for idx in completed_input.split(",") if idx.strip().isdigit()
                            ]
                            for idx in completed_indices:
                                if 0 <= idx < len(steps) and steps[idx] not in completed_steps:
                                    completed_steps.append(steps[idx])
                            st.session_state[selected_goal_title] = completed_steps

                            # Show completed steps and progress
                            st.write("### Completed Steps")
                            for step in completed_steps:
                                st.write(f"- {step}")

                            progress = calculate_goal_progress(steps, completed_steps)
                            st.write(f"Goal Completion Progress: {progress:.2f}%")

                        except ValueError:
                            st.error("Invalid input. Please enter valid step numbers.")

            # Ask the user if they want to continue tracking another goal
            st.write("---")
            continue_choice = st.radio("Would you like to track another goal?", ["Yes", "No"])

            if continue_choice == "Yes":
                # Clear the selected role and goal but retain SMART goals
                st.session_state.pop("selected_role", None)
                st.session_state.pop("selected_goal_title", None)
                st.experimental_rerun()
            else:
                st.write("### Congratulations!")
                st.write("You've made significant progress. Keep up the great work!")

    # Personalized Image Prompt in the Motivational Quotes Section
    elif navigation == "Motivational Quotes":
        with st.spinner("Crafting a personalized career boost message..."):
            st.success("Here's Your Career Power Boost!")

            # Assuming you have a role selected by the user in session_state
            if 'selected_role' in st.session_state:
                selected_role = st.session_state["selected_role"]
                # Personalized motivational quote based on the selected role
                st.write(f"**Your path as a {selected_role} is a journey full of growth, challenges, and incredible opportunities.**")

                # Personalized image based on user's career role
                personalized_prompt = f"A professional and inspiring image of a {selected_role} working in a dynamic environment, symbolizing career growth and success."
                final_image = generate_dynamic_image(personalized_prompt)

                if final_image:
                    st.image(final_image, caption=f"Your Future as a {selected_role} Awaits!", use_column_width=True)
            
            # Action-oriented closing message
            st.write("""
            **Start today. Dream big. Take action now.**
            **Let's make your career extraordinary!**
            """)
    # Add a button to download the full report
    if st.button("Download Full Report as PDF"):
        # Generate the full report
        pdf_file_path = generate_full_report()

        # Provide download link
        with open(pdf_file_path, "rb") as file:
            st.download_button(
                label="Download Full Report",
                data=file,
                file_name="career_report.pdf",
                mime="application/pdf"
            )

# Main Streamlit app entry point
def main():
    st.title("Career Recommendation System")
    st.write("Upload your CV to get personalized career recommendations and refine them based on feedback.")

    uploaded_file = st.file_uploader("Upload your CV (PDF format only):", type=["pdf"])

    if uploaded_file:
        with st.spinner("Processing your CV..."):
            try:
                career_recommendation_engine_with_feedback_and_skill_match_ui(uploaded_file)
            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    app()




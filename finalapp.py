
import tempfile
import os
import fitz  # PyMuPDF
import openai
import streamlit as st



# Set your API key securely (e.g., using environment variables)
openai.api_key = ""  # Or use getpass to input securely






def dynamic_translate(text, target_language):
    """
    Translate text dynamically using OpenAI's ChatGPT API.
    """
    if target_language == "en":  # If the target language is English, return the original text
        return text
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  # Use "gpt-4" if you have access and prefer
            messages=[
                {
                    "role": "system",
                    "content": "You are a multilingual translation assistant. Translate text accurately."
                },
                {
                    "role": "user",
                    "content": f"Translate the following text to {target_language}:\n{text}"
                }
            ],
            temperature=0.2,  # Low temperature for accuracy
        )
        # Extract the translation from the response and strip any extra spaces
        translation = response["choices"][0]["message"]["content"].strip()
        return translation
    except Exception as e:
        return f"[Error Translating]: {e}"






def get_chatgpt_response(prompt, model="gpt-4"):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message['content']




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
    For each role, outline which skills or experiences are strong matches and which may require development. 

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
    

    Job Roles:
    {potential_roles}
    """
    career_path_descriptions = get_chatgpt_response(prompt)
    return career_path_descriptions


def calculate_skill_match(cv_features, refined_recommendations):
    prompt = f"""
    For each of the refined job roles provided, calculate a skill match percentage based on the candidate's skills from their CV.
    Match each skill mentioned in the CV to the skills required for the job role, and return a percentage match score.
    Additionally, provide brief reasoning for the score based on matching and missing skills, but when you give me the refined recommendations and the refined skill match percentages, dont give me the skill match percentages of the old recommendations
    

    CV Skills:
    {cv_features}

    Refined Job Roles:
    {refined_recommendations}
    """

    skill_match_data = get_chatgpt_response(prompt)
    return skill_match_data

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
    -Helpful sources and links

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
    Each SMART goal should include:
    1. A clear and specific description of the goal.
    2. A set of 3–5 actionable, sequential steps to achieve the goal.

    The response should be structured as follows:
    - Each goal should be represented as a tuple where:
    - The first element is a string, describing the goal clearly (Specific, Measurable, Achievable, Relevant, and Time-bound).
    - The second element is a list of 3-5 numbered steps (each step should be actionable and directly related to the goal).
    
    Example format:
    [
        ('Goal 1: Improve technical skills in data science within the next 6 months', [
            'Step 1: Enroll in an online data science course by the end of this month.',
            'Step 2: Complete the first module of the course within 3 weeks.',
            'Step 3: Practice coding daily for 30 minutes to reinforce learning.',
            'Step 4: Join a data science online community for support and networking.'
        ]),
        ('Goal 2: Increase professional network by attending 3 industry events this year', [
            'Step 1: Research and identify 3 relevant industry events within the next week.',
            'Step 2: Register for at least one event by the end of next month.',
            'Step 3: Prepare a personal introduction and professional goals for networking.'
        ])
    ]

    Please ensure the goals are realistic, actionable, and aligned with the career recommendation. Keep the description concise and the steps actionable. Make sure the response is easy to read and well-structured.
    """


    
    # Get the response from ChatGPT
    smart_goals_response = get_chatgpt_response(prompt)

    
    
    return smart_goals_response


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
    st.subheader(dynamic_translate("Feedback on Recommendations", st.session_state["language"]))

    # Rating slider and text area for feedback
    rating = st.slider(
        dynamic_translate("Rate these recommendations (1-10):", st.session_state["language"]),
        1, 10, key="rating_slider"
    )
    feedback = st.text_area(
        dynamic_translate("Your feedback on the recommendations:", st.session_state["language"]),
        key="feedback_area"
    )

    # Check if both rating and feedback are provided before refining
    if st.button(dynamic_translate("Submit Feedback", st.session_state["language"])):
        if rating and feedback.strip():
            st.success(dynamic_translate("Processing your feedback...", st.session_state["language"]))

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
                refined_skill_match = calculate_skill_match(cv_features, refined_recommendations)
                
                # Return the refined data for later use
                return refined_recommendations, refined_skill_match
            except Exception as e:
                st.error(dynamic_translate(f"Error during refinement: {e}", st.session_state["language"]))
        else:
            st.error(dynamic_translate("Please provide both a rating and feedback.", st.session_state["language"]))

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







def generated_smart_goals(refined_recommendation):
    prompt = f"""
    For the career recommendation '{refined_recommendation}', create 2–3 SMART goals.
    Each goal should include:
    - A clear and specific description (Specific, Measurable, Achievable, Relevant, Time-bound).
    - 3–5 sequential steps to achieve the goal, formatted as a numbered list.
    Ensure the output is concise, well-structured, and easy to read.
    When you provide the steps to achieve, I want it to have this header: 'Steps to Achieve:'
    """
    
    # Get the response from ChatGPT
    smart_goals_response = get_chatgpt_response(prompt)
    
    # Parse the response into a structured format (list of goals and steps)
    goals = []
    goal_data = smart_goals_response.split("\n")  # Split response by line

    current_goal = None
    steps = []

    for line in goal_data:
        # Check if line starts with "Goal" to identify the goal title
        if line.lower().startswith("goal"):
            if current_goal:
                goals.append((current_goal, steps))
            current_goal = line.strip()
            steps = []  # Reset steps for the new goal
        elif line.lower().startswith("steps to achieve:"):
            continue  # Skip the "Steps to Achieve:" line
        elif line.strip():  # If it's a non-empty line, it's a step
            steps.append(line.strip())

    if current_goal:
        goals.append((current_goal, steps))  # Add the last goal

    return goals





def interactive_feedback_loop(recommendations, potential_roles, role_analysis, cv_features):
    # Initialize variables
    recommendations = recommendations
    refined_skill_match_data = {}
    iteration = 1  # Track feedback iterations

    print("Welcome to the Interactive Feedback Loop!")
    print("You can provide feedback multiple times. Type 'stop' to end the session.\n")

    while True:
        print(f"\n--- Iteration {iteration}: Current Recommendations ---")
        print(recommendations)
        
        # Collect user feedback
        user_input = input("Enter your feedback or type 'stop' to end: ").strip()
        if user_input.lower() == "stop":
            print("Exiting the feedback loop. Thank you for your input!")
            break
        
        try:
            rating = int(input("On a scale of 1 to 10, how would you rate the recommendations? "))
            if rating < 1 or rating > 10:
                print("Please enter a rating between 1 and 10.")
                continue
        except ValueError:
            print("Please enter a valid integer between 1 and 10.")
            continue

        # Use the feedback to refine recommendations
        feedback = user_input
        prompt = f"""
        Refine the career recommendations based on the following inputs:
        
        1. Current Recommendations:
        {recommendations}
        
        2. User Feedback:
        - Rating: {rating}/10
        - Comments: {feedback}
        
        3. Candidate's CV Features:
        {cv_features}
        
        4. Previous Potential Job Roles:
        {potential_roles}
        
        5. Role Analysis (Strengths and Gaps):
        {role_analysis}
        
        Incorporate the feedback to improve the relevance and alignment of the recommendations.
        """
        
        # Generate refined recommendations using ChatGPT
        refined_recommendations = get_chatgpt_response(prompt)
        
        # Optionally calculate updated skill match data
        refined_skill_match_data = calculate_skill_match(cv_features, potential_roles)
        
        # Update the recommendations for the next iteration
        recommendations = refined_recommendations
        iteration += 1

    return {"refined_recommendations": refined_recommendations, "refined_skill_match_data": refined_skill_match_data}
    

def interactive_recommendation_engine_with_feedback(pdf_path):
    # Step 1: Extract and analyze CV information
    cv_text = extract_text_from_pdf(pdf_path)
    cv_features = get_cv_features(cv_text)
    potential_roles = generate_potential_roles(cv_features)
    role_analysis = analyze_roles_with_cv(cv_features, potential_roles)
    career_path_descriptions = get_career_path_descriptions(potential_roles)

    # Calculate initial skill match percentages
    initial_skill_match_data = calculate_skill_match(cv_features, potential_roles)

    # Prepare initial recommendations
    recommendations = f"""
    Career Recommendations for the Candidate:

    Extracted CV Features:
    {cv_features}

    Recommended Job Roles:
    {potential_roles}

    Role Analysis (Strengths and Gaps):
    {role_analysis}

    Career Path Descriptions (with skill requirements, trends, and growth areas):
    {career_path_descriptions}

    Initial Skill Match Percentages:
    {initial_skill_match_data}
    """
    print("\n--- Initial Career Recommendations ---")
    print(recommendations)

    # Step 2: Start the interactive feedback loop
    user_feedback = interactive_feedback_loop(
        recommendations=recommendations,
        potential_roles=potential_roles,
        role_analysis=role_analysis,
        cv_features=cv_features
    )

def interactive_feedback_loop_ui(recommendations, potential_roles, role_analysis, cv_features):
    # Initialize session state variables
    if "feedback_ongoing" not in st.session_state:
        st.session_state.feedback_ongoing = True  # Feedback loop is active
    if "iteration" not in st.session_state:
        st.session_state.iteration = 1
    if "recommendations" not in st.session_state:
        st.session_state.recommendations = recommendations  # Initial recommendations
    if "skill_match_data" not in st.session_state:
        st.session_state.skill_match_data = None
    if "feedback_history" not in st.session_state:
        st.session_state.feedback_history = []  # Store feedback for each iteration

    # Display recommendations and feedback form
    if st.session_state.feedback_ongoing:
        st.subheader(dynamic_translate(f"Iteration {st.session_state.iteration}: Refined Recommendations", st.session_state["language"]))
        st.text_area(
            dynamic_translate(f"Recommendations for Iteration {st.session_state.iteration}", st.session_state["language"]),
            value=dynamic_translate(st.session_state.recommendations, st.session_state["language"]),
            height=200,
            disabled=True,
            key=f"recommendations_{st.session_state.iteration}",
        )

        # Provide feedback for the current iteration
        st.subheader(dynamic_translate(f"Provide Your Feedback for Iteration {st.session_state.iteration}", st.session_state["language"]))
        feedback = st.text_area(
            dynamic_translate(f"Enter your feedback for iteration {st.session_state.iteration}:", st.session_state["language"]),
            key=f"feedback_input_{st.session_state.iteration}",
        )
        rating = st.slider(
            dynamic_translate(f"Rate the recommendations for iteration {st.session_state.iteration} (1-10):", st.session_state["language"]),
            1,
            10,
            key=f"rating_slider_{st.session_state.iteration}",
        )

        # Submit feedback button
        if st.button(dynamic_translate(f"Submit Feedback for Iteration {st.session_state.iteration}", st.session_state["language"])):
            if not feedback.strip():
                st.error(dynamic_translate("Feedback cannot be empty.", st.session_state["language"]))
            else:
                st.success(dynamic_translate("Processing feedback...", st.session_state["language"]))
                # Generate prompt and refine recommendations
                try:
                    prompt = f"""
                    Refine recommendations based on:
                    - Feedback: {feedback}
                    - Rating: {rating}/10
                    - Current Recommendations: {st.session_state.recommendations}
                    """
                    refined_recommendations = dynamic_translate(get_chatgpt_response(prompt), st.session_state["language"])
                    st.session_state.recommendations = refined_recommendations
                    st.session_state.iteration += 1
                    st.rerun()  # Refresh UI for the next iteration
                except Exception as e:
                    st.error(f"An error occurred: {e}")

    # End Feedback Loop Button
    if st.button(dynamic_translate("End Feedback Loop", st.session_state["language"])):
        st.session_state.feedback_ongoing = False  # Stop feedback loop
        st.session_state.final_recommendations = st.session_state.recommendations  # Store final recommendations

        st.subheader(dynamic_translate("Final Recommendations", st.session_state))
        st.text_area(
            dynamic_translate("Final Recommendations", st.session_state["language"]),
            value=st.session_state.recommendations,
            height=200,
            disabled=True,
        )
        st.write(dynamic_translate("Thank you for your feedback!", st.session_state["language"]))
        st.write(dynamic_translate("Feedback History:", st.session_state["language"]))
        for feedback in st.session_state.feedback_history:
            st.write(feedback)
        st.stop()


    return st.session_state.recommendations, st.session_state.skill_match_data 






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

    # Language selection
    LANGUAGES = {
        "en": "English",
        "fr": "French",
        "es": "Spanish",
        "ar": "Arabic",
    }

    if "language" not in st.session_state:
        st.session_state["language"] = "en"  # Default to English

    selected_language = st.selectbox(
        "Choose Language:",
        options=list(LANGUAGES.keys()),
        format_func=lambda x: LANGUAGES[x],
        key="language_selector"
    )
    st.session_state["language"] = selected_language

    # Store translations in session state based on the selected language
    if "translations" not in st.session_state:
        st.session_state["translations"] = {
            "choose_section": dynamic_translate("Choose a section:", st.session_state["language"]),
            "home": dynamic_translate("Home", st.session_state["language"]),
            "upload": dynamic_translate("Upload CV", st.session_state["language"]),
            "view": dynamic_translate("View Recommendations", st.session_state["language"]),
            "feedB": dynamic_translate("Feedback & Refinement", st.session_state["language"]),
            "generate": dynamic_translate("Generate and Track SMART Goals", st.session_state["language"]),
        }

    # Update the navigation keys with the translated strings directly
    navigation_keys = {
        "home": st.session_state["translations"]["home"],
        "upload": st.session_state["translations"]["upload"],
        "view": st.session_state["translations"]["view"],
        "feedB": st.session_state["translations"]["feedB"],
        "generate": st.session_state["translations"]["generate"],
    }

    # Sidebar
    if "navigation" not in st.session_state:
        st.session_state["navigation"] = "home"

    # Use translations in the sidebar radio button
    selected_key = st.sidebar.radio(
        st.session_state["translations"]["choose_section"],
        options=list(navigation_keys.keys()),  # Use unique keys for selection
        format_func=lambda key: navigation_keys[key],  # Display translations in the sidebar
    )

    st.session_state["navigation"] = selected_key


    # Home Page with Motivational Quote and Image
    if st.session_state["navigation"]  == "home":
        
        st.title(dynamic_translate("Welcome to the Career Recommendation System!", st.session_state["language"]))
        st.subheader(dynamic_translate("Your journey towards a better career starts here!", st.session_state["language"]))
        
        motivational_quote = dynamic_translate("Success is the sum of small efforts, repeated day in and day out.", st.session_state["language"])
        st.write(motivational_quote)
        
        st.image("https://i.imgur.com/FZSPFtJ.jpeg", caption=dynamic_translate("Your Future Awaits", st.session_state["language"]), use_container_width=True)

    # Upload CV Section
    elif st.session_state["navigation"] == "upload":
        st.title(dynamic_translate("Upload Your CV", st.session_state["language"]))

        uploaded_file = st.file_uploader(
            dynamic_translate("Upload your CV (PDF format)", st.session_state["language"]), 
            type=["pdf"]
        )

        if uploaded_file:
            # Check if CV features are already present in session state
            if "cv_features" not in st.session_state:
                with st.spinner(dynamic_translate("Extracting CV text...", st.session_state["language"])):
                    cv_text = dynamic_translate(extract_text_from_pdf(uploaded_file), st.session_state["language"])
                
                # Save extracted text in session state
                st.session_state["cv_text"] = cv_text
                st.session_state["uploaded_file"] = uploaded_file

                st.success(dynamic_translate("CV text extracted successfully!", st.session_state["language"]))
                st.text_area(dynamic_translate("Extracted CV Text", st.session_state["language"]), cv_text, height=400)

                # Extract features from CV text and store in session state
                with st.spinner(dynamic_translate("Extracting CV features...", st.session_state["language"])):
                    cv_features = get_cv_features(cv_text)
                st.session_state["cv_features"] = cv_features
            else:
                st.info(dynamic_translate("CV features already exist in the session!", st.session_state["language"]))
        else:
            st.warning(dynamic_translate("Please upload your CV first!", st.session_state["language"]))


   # View Recommendations Section
    elif st.session_state["navigation"] == "view":
        st.title(dynamic_translate("Career Recommendations", st.session_state["language"]))

        if "cv_text" in st.session_state:
            # Extract CV features only if not already present
            if "cv_features" not in st.session_state:
                with st.spinner(dynamic_translate("Analyzing CV features...", st.session_state["language"])):
                    cv_features = get_cv_features(st.session_state["cv_text"])
                st.session_state["cv_features"] = cv_features
            else:
                cv_features = st.session_state["cv_features"]

            # Generate potential roles only if not already present
            if "potential_roles" not in st.session_state:
                with st.spinner(dynamic_translate("Generating potential job roles...", st.session_state["language"])):
                    potential_roles = generate_potential_roles(cv_features)
                st.session_state["potential_roles"] = potential_roles
            else:
                potential_roles = st.session_state["potential_roles"]

            st.subheader(dynamic_translate("Potential Job Roles", st.session_state["language"]))
            st.write(dynamic_translate(potential_roles, st.session_state["language"]))

            # Analyze roles only if not already present
            if "role_analysis" not in st.session_state:
                with st.spinner(dynamic_translate("Analyzing roles with CV...", st.session_state["language"])):
                    role_analysis = analyze_roles_with_cv(cv_features, potential_roles)
                st.session_state["role_analysis"] = role_analysis
                st.write(dynamic_translate(role_analysis, st.session_state["language"]))
        else:
            st.warning(dynamic_translate("Please upload your CV first!", st.session_state["language"]))


    # Feedback & Refinement Section
    elif st.session_state["navigation"]  == "feedB":
        st.title(dynamic_translate("Feedback & Refine Recommendations", st.session_state["language"]))
        
        if "cv_text" in st.session_state:
            if "cv_features" not in st.session_state or "potential_roles" not in st.session_state:
                st.warning(dynamic_translate("Please generate recommendations first in the 'View Recommendations' section.", st.session_state["language"]))
            else:
                potential_roles = st.session_state["potential_roles"]
                cv_features = st.session_state["cv_features"]
                if "role_analysis" not in st.session_state:
                    with st.spinner(dynamic_translate("Analyzing roles with CV...", st.session_state["language"])):
                        st.session_state["role_analysis"] = analyze_roles_with_cv(cv_features, potential_roles)
                role_analysis = st.session_state["role_analysis"]
                

                refined_recommendations, refined_skill_match_data = interactive_feedback_loop_ui(potential_roles,potential_roles, role_analysis, cv_features)

                if refined_recommendations and refined_skill_match_data:
                    st.success(dynamic_translate("Recommendations refined successfully!", st.session_state["language"]))
                    st.subheader(dynamic_translate("Skill Match Percentages", st.session_state["language"]))
                    st.write(dynamic_translate(refined_skill_match_data, st.session_state["language"]))
                    st.session_state["refined_recommendations"] = refined_recommendations
                    st.session_state["refined_skill_match_data"] = refined_skill_match_data
                else:
                    st.warning(dynamic_translate("Please provide feedback to refine the recommendations.", st.session_state["language"]))
        else:
            st.warning(dynamic_translate("Please upload your CV first!", st.session_state["language"]))

        
   # Generate and Track SMART Goals Section
    elif st.session_state["navigation"]  == "generate":
        # Generate and Track SMART Goals Interface
        st.title(dynamic_translate("Generate and Track SMART Goals", st.session_state["language"]))
      

        # Check if refined recommendations exist
        if "final_recommendations" not in st.session_state or not st.session_state["final_recommendations"]:
            st.warning(dynamic_translate("Please refine recommendations first in the 'Feedback & Refinement' section.", st.session_state["language"]))
        else:
            st.write(dynamic_translate("Based on your final recommendations, let's generate SMART goals.", st.session_state["language"]))
            st.text_area(
                dynamic_translate("Final Recommendations", st.session_state["language"]),
                value=st.session_state.final_recommendations,
                height=200,
                disabled=True,
            )
            refined_recommendations = st.session_state["final_recommendations"]

            # Ensure only career titles are extracted
            career_titles = extract_career_titles(refined_recommendations)
            
            st.write(career_titles)
            
            #st.write(career_titles)
            st.session_state["career_titles"] = career_titles
            
            # Initialize smart_goals if not already done
            if "smart_goals" not in st.session_state:
                st.session_state["smart_goals"] = {}
                for title in career_titles:
                    st.write(title)
                    with st.spinner(dynamic_translate(f"Generating SMART goals for {title}...", st.session_state["language"])):
                        # Generate the SMART goals
                        st.session_state["smart_goals"][title] = generated_smart_goals(title)
                        #st.write(st.session_state["smart_goals"][title])

            # Display SMART goals for all career paths before the dropdown
            st.subheader(dynamic_translate("SMART Goals for All Career Paths", st.session_state["language"]))
            for path, goals in st.session_state["smart_goals"].items():
                # Translate the career path name
                st.write(f"### {dynamic_translate(path, st.session_state['language'])}")
                if goals:
                    for idx, (goal, steps) in enumerate(goals, 1):
                        st.write(f" {idx}: {dynamic_translate(goal, st.session_state['language'])}")
                        for step in steps:
                            st.markdown(f"- {dynamic_translate(step, st.session_state['language'])}")
                else:
                    st.write(dynamic_translate("No SMART goals available.", st.session_state["language"]))

            # For displaying the goals for the selected career path
            # Ensure dynamic_translate is applied to each item in career_titles
            translated_career_titles = [dynamic_translate(title, st.session_state["language"]) for title in career_titles]

            # Combine the translated career titles with the default option
            options = translated_career_titles

            # Display dropdown with translated options
            selected_path = st.selectbox(
                dynamic_translate("Select a career recommendation to view its SMART goals:", st.session_state["language"]),
                options,
                key="selected_path"
            )


            if selected_path == dynamic_translate("Select a career recommendation", st.session_state["language"]):
                st.info(dynamic_translate("Please select a Career Recommendation to view its SMART goals.", st.session_state["language"]))
            else:
                # Check if goals are available for the selected path
                if selected_path not in st.session_state["smart_goals"] or not st.session_state["smart_goals"][selected_path]:
                    st.warning(dynamic_translate(f"No SMART goals found for {selected_path}. Please generate the goals first.", st.session_state["language"]))
                else:
                    raw_goals = st.session_state["smart_goals"].get(selected_path, [])

                    # Initialize the goal dictionary
                    goal_dict = {}

                    if raw_goals:
                        for idx, (goal, steps) in enumerate(raw_goals, 1):
                            goal_dict[idx] = (goal, steps)
                            st.write(f" {idx}: {dynamic_translate(goal, st.session_state['language'])}")
                            for step in steps:
                                st.markdown(f"- {dynamic_translate(step, st.session_state['language'])}")

                        # Select goal to track progress
                        selected_goal_index = st.selectbox(
                            dynamic_translate("Select a goal to track progress:", st.session_state["language"]),
                            [dynamic_translate("Select Goal", st.session_state["language"])] + list(range(1, len(goal_dict) + 1)),
                            key=f"select_goal_{selected_path}"
                        )

                        if selected_goal_index == dynamic_translate("Select Goal", st.session_state["language"]):
                            st.info(dynamic_translate("Please select a goal to view and track its progress.", st.session_state["language"]))
                        elif selected_goal_index is not None:
                            selected_goal_title, steps = goal_dict[selected_goal_index]
                            goal_key = f"{selected_path} - Goal {selected_goal_index}"

                            # Initialize goal progress if not done
                            if "goal_progress" not in st.session_state:
                                st.session_state["goal_progress"] = {}

                            if goal_key not in st.session_state["goal_progress"]:
                                st.session_state["goal_progress"][goal_key] = []

                            completed_steps = st.session_state["goal_progress"][goal_key]

                            # Display the goal title and steps before tracking progress
                            st.write(f"**{dynamic_translate('Goal Title:', st.session_state['language'])}** {dynamic_translate(selected_goal_title, st.session_state['language'])}")
                            st.write(dynamic_translate("**Steps to Complete this Goal:**", st.session_state["language"]))
                            for step in steps:
                                st.markdown(f"- {dynamic_translate(step, st.session_state['language'])}")

                            # Track progress UI
                            st.subheader(dynamic_translate(f"Track Progress for Goal: {selected_goal_title}", st.session_state["language"]))
                            completed_input = st.text_input(dynamic_translate("Enter the numbers of completed steps (comma-separated):", st.session_state["language"]))

                            if st.button(dynamic_translate("Submit", st.session_state["language"])):
                                if completed_input:
                                    try:
                                        # Process the completed step indices
                                        completed_indices = [
                                            int(idx.strip()) - 1 for idx in completed_input.split(",")
                                            if idx.strip().isdigit()
                                        ]

                                        # Extract steps that match the numbering pattern
                                        numbered_steps = [
                                            step for step in steps if step.strip().split()[0].rstrip('.').isdigit()
                                        ]

                                        if not numbered_steps:
                                            st.error(dynamic_translate("No valid numbered steps found. Please check your step formatting.", st.session_state["language"]))
                                            return

                                        new_steps = [
                                            numbered_steps[idx] for idx in completed_indices
                                            if 0 <= idx < len(numbered_steps) and numbered_steps[idx] not in completed_steps
                                        ]
                                        completed_steps.extend(new_steps)
                                        st.session_state["goal_progress"][goal_key] = completed_steps

                                        # Display completed steps
                                        st.write(dynamic_translate("### Completed Steps", st.session_state["language"]))
                                        for step in completed_steps:
                                            st.write(f"- {dynamic_translate(step, st.session_state['language'])}")

                                        # Calculate progress
                                        progress = len(completed_steps) / len(numbered_steps) * 100
                                        st.write(f"{dynamic_translate('Goal Completion Progress:', st.session_state['language'])} {progress:.2f}%")

                                        if progress < 100:
                                            incomplete_steps = [step for step in numbered_steps if step not in completed_steps]
                                            st.write(dynamic_translate("Keep going! You're doing great.", st.session_state["language"]))
                                            st.write(dynamic_translate("### Resources for Completing Remaining Steps:", st.session_state["language"]))
                                            resource_prompt = dynamic_translate(f"Provide motivational advice and resources for the following incomplete steps: {incomplete_steps}", st.session_state["language"])
                                            motivational_feedback = get_chatgpt_response(resource_prompt)
                                            st.write(dynamic_translate(motivational_feedback, st.session_state["language"]))
                                        else:
                                            st.write(dynamic_translate("You completed this goal!", st.session_state["language"]))

                                    except ValueError:
                                        st.error(dynamic_translate("Invalid input. Please enter step numbers as comma-separated integers.", st.session_state["language"]))
                                else:
                                    st.warning(dynamic_translate("Please enter step numbers before submitting.", st.session_state["language"]))
                    else:
                            st.warning(f"No goals found for {selected_path}. Please generate SMART goals first.")

            if st.button(dynamic_translate("Finish", st.session_state["language"])):
                # Display congratulatory messages
                st.write(dynamic_translate("### Congratulations!", st.session_state["language"]))
                st.write(dynamic_translate("You've taken the first step towards achieving your goals.", st.session_state["language"]))
                # Fetch and display a motivational quote
                quote = get_motivational_quote()
                st.write(dynamic_translate(f"Here's some inspiration for you: {quote}", st.session_state["language"]))
                # Exit or stop further execution (optional, based on your app's flow)
                st.stop()  # This stops Streamlit from executing further after the Finish button is pressed



    




if __name__ == "__main__":
    app()





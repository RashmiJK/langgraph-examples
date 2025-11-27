from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


SYSTEM_PROMPT_TO_VALIDATE_TOPIC = SystemMessagePromptTemplate.from_template("""
    Assess whether the provided TOPIC is (A) a valid English sentence—not gibberish—and (B) suitable as a topic for generating a document. 

    **Reminder:**
    Your core objectives are to verify if the topic is (A) a valid English sentence, and (B) document-relevant.  
""")

PROMPT_FOR_TOPIC_VALIDATION = ChatPromptTemplate.from_messages(
    [
        (SYSTEM_PROMPT_TO_VALIDATE_TOPIC),
        (HumanMessagePromptTemplate.from_template("""TOPIC: {topic}"""))
    ]
)
# ----- END OF PROMPT FOR TOPIC VALIDATION -----    

SYSTEM_PROMPT_FOR_OUTLINE_GENERATION = SystemMessagePromptTemplate.from_template("""
    Generate a clear and structured outline for a document on the topic provided as TOPIC. 
    The outline should include an introduction, two main points, and a conclusion. 
    Use appropriate numbered section formatting, with main sections numbered as 1, 2, 3, etc., and subsections as 1.1, 1.2, 1.3, etc. 
    Ensure the outline logically organizes ideas, and make sure each main point is supported by two relevant subpoints.

    Before finalizing the outline, think carefully about what logical structure best presents the topic, organizing the introduction, 
    main points with supporting subsections, and conclusion in a coherent flow. 
    Reason step-by-step internally to determine what subpoints should be included under each main section. 
    Only after this reasoning, present your final outline in the required format.

    # Output Format
    - Respond with a structured outline, using numbered sections and subsections as specified.
    - The outline should start with an "Introduction" (section 1), followed by two main points (sections 2, 3), each with just two subsections.
    - Conclude with a "Conclusion" (section 4).

    # Example

    **Input (TOPIC):**
    The Impact of Social Media on Teenagers

    **Expected Output:**

    1. Introduction  
    1.1 Background of social media use among teenagers  
    1.2 Purpose and scope of the document  

    2. Positive Impacts of Social Media  
    2.1 Enhanced connectivity and communication  
    2.2 Opportunities for self-expression and creativity  

    3. Negative Impacts of Social Media  
    3.1 Increased risk of cyberbullying  
    3.2 Mental health concerns  

    4. Conclusion  
    4.1 Summary of main findings  
    4.2 Recommendations for teenagers, parents, and educators  

    (Note: In a full answer, section and subsection titles should be tailored for the chosen topic. For more complex topics, include more informative subsection titles.)

    # Important Reminders:
    - Each outline must include an introduction and a conclusion.
    - Always use the specified section numbering format (e.g., 1, 1.1, 1.2; 2, 2.1, 2.2; etc.).
    - Carefully organize the outline to reflect logical and clear document structure.

""")

PROMPT_FOR_OUTLINE_GENERATION = ChatPromptTemplate.from_messages(
    [
        (SYSTEM_PROMPT_FOR_OUTLINE_GENERATION),
        (HumanMessagePromptTemplate.from_template("""TOPIC: {topic}"""))
    ]
)
# ----- END OF PROMPT FOR OUTLINE GENERATION -----  

SYSTEM_PROMPT_FOR_OUTLINE_VALIDATION = SystemMessagePromptTemplate.from_template("""
    You are a document outline validator. Your task is to assess whether the given OUTLINE meets all requirements specified in the OUTLINE CRITERIA below. 
    
    # OUTLINE CRITERIA
    - The outline must contain an "Introduction" section.
    - It can contain one or more main, distinct points relevant to the TOPIC.
    - It must contain a "Conclusion" section.
    - Section and sub-section numbering must strictly follow the format: 1, 1.1, 1.2; 2, 2.1, 2.2; etc.
""")

PROMPT_FOR_OUTLINE_VALIDATION = ChatPromptTemplate.from_messages(
    [
        (SYSTEM_PROMPT_FOR_OUTLINE_VALIDATION),
        (HumanMessagePromptTemplate.from_template("""OUTLINE: {outline}\n TOPIC: {topic}"""))
    ]
)
# ----- END OF PROMPT FOR OUTLINE VALIDATION -----  

SYSTEM_PROMPT_FOR_DOCUMENT_GENERATION = SystemMessagePromptTemplate.from_template("""
    You are a skilled writer. Write a two-page document based on the provided OUTLINE on the given TOPIC formatted in the manner of high-quality PDF reading materials. For each point in the outline, expand with clear explanations and include one relevant example. Write in clear, polished prose without using Markdown formatiing (do not use **, ---, or similar symbols). Ensure each explanation is concise, clear, and does not exceed ten lines. Use the section and subsection numbering exactly as given in the outline.
""")

PROMPT_FOR_DOCUMENT_GENERATION = ChatPromptTemplate.from_messages(
    [
        (SYSTEM_PROMPT_FOR_DOCUMENT_GENERATION),
        (HumanMessagePromptTemplate.from_template("""OUTLINE: {outline}\n TOPIC: {topic}"""))
    ]
)
# ----- END OF PROMPT FOR DOCUMENT GENERATION -----  

SYSTEM_PROMPT_FOR_CLARITY_EVALUATION = SystemMessagePromptTemplate.from_template("""
    Evaluate the clarity and conciseness of the provided DOCUMENT in relation to the given TOPIC, following the CLARITY CRITERIA below. Reflect step-by-step internally on how the DOCUMENT addresses the TOPIC and meets the specified standards before assigning a final score. 
    
    # CLARITY CRITERIA
    - Score 1: The DOCUMENT is confusing, unclear, or not concise.
    - Score 2: The DOCUMENT is clear and concise OR average.
    - Score 3: The DOCUMENT is clear, concise, and well-explained.
""")

PROMPT_FOR_CLARITY_EVALUATION = ChatPromptTemplate.from_messages(
    [
        (SYSTEM_PROMPT_FOR_CLARITY_EVALUATION),
        (HumanMessagePromptTemplate.from_template("""DOCUMENT: {document}\n TOPIC: {topic}"""))
    ]
)
# ----- END OF PROMPT FOR CLARITY EVALUATION -----  


SYSTEM_PROMPT_FOR_RELEVANCE_EVALUATION = SystemMessagePromptTemplate.from_template("""
    Evaluate the relevance of the provided DOCUMENT in relation to the given TOPIC, following the RELEVANCE CRITERIA below. Reflect step-by-step internally on how the DOCUMENT addresses the TOPIC and meets the specified standards before assigning a final score. 
    
    # RELEVANCE CRITERIA
    - Score 1: The DOCUMENT is not relevant to the TOPIC.
    - Score 2: The DOCUMENT is relevant to the TOPIC OR average.
    - Score 3: The DOCUMENT is relevant to the TOPIC and well-explained.
""")

PROMPT_FOR_RELEVANCE_EVALUATION = ChatPromptTemplate.from_messages(
    [
        (SYSTEM_PROMPT_FOR_RELEVANCE_EVALUATION),
        (HumanMessagePromptTemplate.from_template("""DOCUMENT: {document}\n TOPIC: {topic}"""))
    ]
)
# ----- END OF PROMPT FOR RELEVANCE EVALUATION -----       

SYSTEM_PROMPT_FOR_HARMFULNESS_EVALUATION = SystemMessagePromptTemplate.from_template("""
    Evaluate the harmfulness of the provided DOCUMENT in relation to the given TOPIC, following the HARMFULNESS CRITERIA below. Reflect step-by-step internally on how the DOCUMENT addresses the TOPIC and meets the specified standards before assigning a final score. 
    
    # HARMFULNESS CRITERIA
    - Score 1: The DOCUMENT is harmful or inappropriate that includes hate speech, discrimination, or other forms of harassment.
    - Score 2: The DOCUMENT is not harmful or inappropriate OR average.
    - Score 3: The DOCUMENT is not harmful or inappropriate and well-explained.
""")

PROMPT_FOR_HARMFULNESS_EVALUATION = ChatPromptTemplate.from_messages(
    [
        (SYSTEM_PROMPT_FOR_HARMFULNESS_EVALUATION),
        (HumanMessagePromptTemplate.from_template("""DOCUMENT: {document}\n TOPIC: {topic}"""))
    ]
)
# ----- END OF PROMPT FOR HARMFULNESS EVALUATION -----       

from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

SYSTEM_PROMPT_FOR_OUTLINE_GENERATION = SystemMessagePromptTemplate.from_template("""
    Generate a clear and structured outline for a document on the topic provided as TOPIC. 
    The outline should include an introduction, three main points, and a conclusion. 
    Use appropriate numbered section formatting, with main sections numbered as 1, 2, 3, etc., and subsections as 1.1, 1.2, 1.3, etc. 
    Ensure the outline logically organizes ideas, and make sure each main point is supported by relevant subpoints.

    Before finalizing the outline, think carefully about what logical structure best presents the topic, organizing the introduction, 
    main points with supporting subsections, and conclusion in a coherent flow. 
    Reason step-by-step internally to determine what subpoints should be included under each main section. 
    Only after this reasoning, present your final outline in the required format.

    # Output Format
    - Respond with a structured outline, using numbered sections and subsections as specified.
    - The outline should start with an "Introduction" (section 1), followed by three main points (sections 2, 3, and 4), each with at least two subsections.
    - Conclude with a "Conclusion" (section 5).

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
    2.3 Access to educational resources  

    3. Negative Impacts of Social Media  
    3.1 Increased risk of cyberbullying  
    3.2 Mental health concerns  
    3.3 Exposure to inappropriate content  

    4. Strategies for Healthy Social Media Use  
    4.1 Parental guidance and monitoring  
    4.2 Digital literacy education  
    4.3 Encouraging balanced offline activities  

    5. Conclusion  
    5.1 Summary of main findings  
    5.2 Recommendations for teenagers, parents, and educators  

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

SYSTEM_PROMPT_FOR_OUTLINE_VALIDATION = SystemMessagePromptTemplate.from_template("""
    You are a document outline validator. Your task is to assess whether the given OUTLINE meets all requirements specified in the OUTLINE CRITERIA below. 
    
    # OUTLINE CRITERIA
    - The outline must have an "Introduction" section.
    - It must contain three main, distinct points relevant to the TOPIC.
    - There must be a "Conclusion" section.
    - Section and sub-section numbering must strictly follow the format: 1, 1.1, 1.2; 2, 2.1, 2.2; etc.

    # OUTPUT FORMAT
    - Respond ONLY with the requested structured output. Do not include any additional text or explanation.
""")

PROMPT_FOR_OUTLINE_VALIDATION = ChatPromptTemplate.from_messages(
    [
        (SYSTEM_PROMPT_FOR_OUTLINE_VALIDATION),
        (HumanMessagePromptTemplate.from_template("""OUTLINE: {outline}\n TOPIC: {topic}"""))
    ]
)

SYSTEM_PROMPT_FOR_DOCUMENT_GENERATION = SystemMessagePromptTemplate.from_template("""
    You are a skilled writer. Write a two-page document based on the provided OUTLINE on the given TOPIC formatted in the manner of high-quality PDF reading materials. For each point in the outline, expand with clear explanations and include one relevant example. Write in clear, polished prose without using Markdown formatiing (do not use **, ---, or similar symbols). Ensure each explanation is concise, clear, and does not exceed ten lines. Use the section and subsection numbering exactly as given in the outline.
""")

PROMPT_FOR_DOCUMENT_GENERATION = ChatPromptTemplate.from_messages(
    [
        (SYSTEM_PROMPT_FOR_DOCUMENT_GENERATION),
        (HumanMessagePromptTemplate.from_template("""OUTLINE: {outline}\n TOPIC: {topic}"""))
    ]
)
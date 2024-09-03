import streamlit as st
from transformers import pipeline
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize the question-answering pipeline using a pretrained model
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Sample dataset of questions and answers
data = {
  "data": [
    {
      "question": "What is IBM Rational Rhapsody?",
      "answer": "IBM Rational Rhapsody is a visual development environment used for modeling, designing, and generating code for embedded and real-time systems. It supports modeling languages like UML and SysML."
    },
    {
      "question": "How do I create a new project in Rhapsody?",
      "answer": "To create a new project in Rhapsody, go to the File menu, select 'New', and then choose 'Project'. Follow the prompts to set up your project name, location, and initial configuration."
    },
    {
      "question": "What programming languages does Rhapsody support?",
      "answer": "Rhapsody supports multiple programming languages, including C, C++, Java, Ada, and C#."
    },
    {
      "question": "How do I generate code in Rhapsody?",
      "answer": "To generate code in Rhapsody, ensure your model is complete and then select 'Generate Code' from the 'Code' menu. You can choose specific elements or the entire model for code generation."
    },
    {
      "question": "What is model-driven development in Rhapsody?",
      "answer": "Model-driven development (MDD) in Rhapsody involves using visual models to define system requirements, architecture, and behavior. The models are then used to automatically generate code, reducing development time and errors."
    },
    {
      "question": "Can I integrate Rhapsody with other tools?",
      "answer": "Yes, Rhapsody integrates with various tools, including configuration management tools (e.g., Git, SVN), testing tools, and other IBM Rational products like DOORS and Team Concert."
    },
    {
      "question": "What is SysML, and how is it used in Rhapsody?",
      "answer": "SysML (Systems Modeling Language) is a general-purpose modeling language used for system engineering applications. In Rhapsody, SysML can be used to create models that describe system requirements, behavior, and architecture."
    },
    {
      "question": "What is the role of UML in Rhapsody?",
      "answer": "UML (Unified Modeling Language) is a standardized modeling language used to specify, visualize, construct, and document the artifacts of software systems. In Rhapsody, UML is used to create various types of diagrams, such as class diagrams, sequence diagrams, and state charts, to model different aspects of a system."
    },
    {
      "question": "What are the main features of Rhapsody?",
      "answer": "Rhapsody offers features like model-driven development, real-time system modeling, automatic code generation, support for UML and SysML, and integration with other development tools and environments."
    },
    {
      "question": "How do I use the Rhapsody API?",
      "answer": "The Rhapsody API allows developers to customize and automate tasks within Rhapsody. To use the API, you must enable scripting support, and then you can write scripts in JavaScript or other supported languages to interact with the Rhapsody environment."
    },
    {
      "question": "How can I create a state machine in Rhapsody?",
      "answer": "To create a state machine in Rhapsody, select the 'Statechart' option from the model elements, and then define states, transitions, and actions using the graphical editor."
    },
    {
      "question": "What are the system requirements for installing IBM Rational Rhapsody?",
      "answer": "The system requirements for installing IBM Rational Rhapsody vary by version but typically include a Windows or Linux operating system, a certain amount of RAM (usually 4GB or more), and available disk space. It's also important to have a compatible Java Runtime Environment (JRE) installed."
    },
    {
      "question": "How do I simulate a model in Rhapsody?",
      "answer": "To simulate a model in Rhapsody, use the 'Simulation' menu, where you can select 'Run' to execute the model and observe its behavior in a controlled environment."
    },
    {
      "question": "How do I customize the Rhapsody environment?",
      "answer": "You can customize the Rhapsody environment by adjusting settings in the 'Tools' menu under 'Options', modifying toolbars and menus, or using the Rhapsody API to automate repetitive tasks and integrate with other tools."
    },
    {
      "question": "What is the Rhapsody Gateway?",
      "answer": "The Rhapsody Gateway is a tool that allows users to synchronize data between IBM Rational Rhapsody and other tools, such as IBM Rational DOORS. It helps in maintaining consistency between requirements and models."
    },
    {
      "question": "How do I set up version control in Rhapsody?",
      "answer": "To set up version control in Rhapsody, use the 'Configuration Management' options in the 'Tools' menu. You can integrate with tools like Git or SVN for version control."
    },
    {
      "question": "Can Rhapsody be used for safety-critical systems?",
      "answer": "Yes, Rhapsody is often used for developing safety-critical systems, such as those in the automotive, aerospace, and medical industries. It supports compliance with industry standards like DO-178C and ISO 26262."
    },
    {
      "question": "How do I manage requirements in Rhapsody?",
      "answer": "You can manage requirements in Rhapsody by linking them to model elements, using the Requirements view to trace and validate them, and integrating with IBM Rational DOORS for advanced requirements management."
    },
    {
      "question": "What is the difference between Rhapsody Designer and Rhapsody Architect?",
      "answer": "Rhapsody Designer is a full-featured edition with support for model-driven development and code generation, while Rhapsody Architect is a lighter version focused on design and modeling without code generation capabilities."
    },
    {
      "question": "How do I perform unit testing in Rhapsody?",
      "answer": "Rhapsody provides tools for model-based testing, including the ability to create test cases, simulate scenarios, and generate test scripts for unit testing purposes."
    },
    {
      "question": "How do I use profiles in Rhapsody?",
      "answer": "Profiles in Rhapsody are used to extend UML with custom stereotypes, tagged values, and constraints, allowing for domain-specific modeling and analysis."
    },
    {
      "question": "What is a stereotype in Rhapsody?",
      "answer": "A stereotype in Rhapsody is a custom extension of a UML model element that defines additional properties and behaviors, allowing for more specialized modeling."
    },
    {
      "question": "How can I collaborate with other users in Rhapsody?",
      "answer": "You can collaborate with other users in Rhapsody by using version control systems, shared repositories, or by utilizing Rhapsody's team collaboration features, such as commenting and reviewing changes."
    },
    {
      "question": "What is a sequence diagram in Rhapsody?",
      "answer": "A sequence diagram in Rhapsody is a type of UML diagram that shows how objects interact in a particular sequence of events, useful for detailing the flow of control and data in a system."
    },
    {
      "question": "How do I create a class diagram in Rhapsody?",
      "answer": "To create a class diagram in Rhapsody, select the 'Class Diagram' option from the model elements and then drag and drop classes, associations, and other elements to define the structure of your system."
    },
    {
      "question": "How can I automate tasks in Rhapsody?",
      "answer": "You can automate tasks in Rhapsody by using the scripting interface, writing scripts in languages like JavaScript or VBScript to interact with the tool's API and automate repetitive actions."
    },
    {
      "question": "What is real-time analysis in Rhapsody?",
      "answer": "Real-time analysis in Rhapsody involves examining the timing and performance characteristics of a system to ensure it meets the real-time requirements, such as response times and deadlines."
    },
    {
      "question": "How do I perform model validation in Rhapsody?",
      "answer": "To perform model validation in Rhapsody, use the 'Check Model' feature, which examines the model for errors, inconsistencies, and adherence to defined rules and guidelines."
    },
    {
      "question": "What are the benefits of using Rhapsody for embedded systems development?",
      "answer": "Rhapsody offers benefits like reduced development time, improved code quality, automatic code generation, model-based testing, and support for industry standards in embedded systems development."
    },
    {
      "question": "How can I create a use case diagram in Rhapsody?",
      "answer": "To create a use case diagram in Rhapsody, select the 'Use Case Diagram' option and then define actors, use cases, and relationships to represent the functional requirements of a system."
    },
    {
      "question": "How do I debug a model in Rhapsody?",
      "answer": "To debug a model in Rhapsody, use the built-in debugging tools to step through the model execution, inspect variables, and analyze the flow of control to identify and fix issues."
    },
    {
      "question": "What is model animation in Rhapsody?",
      "answer": "Model animation in Rhapsody allows you to visualize the execution of a model in real-time, showing the sequence of events, state changes, and interactions between components."
    },
    {
      "question": "How do I export a model from Rhapsody?",
      "answer": "To export a model from Rhapsody, use the 'Export' option in the File menu, where you can choose formats like XMI, HTML, or image files for sharing or documentation purposes."
    },
    {
      "question": "Can Rhapsody be used for Agile development?",
      "answer": "Yes, Rhapsody can be used in Agile development environments by supporting iterative development, continuous integration, and collaborative modeling practices."
    },
    {
      "question": "How do I add constraints in Rhapsody?",
      "answer": "To add constraints in Rhapsody, use the 'Constraint' option from the model elements and attach it to a model element to specify rules and conditions that must be met."
    },
    {
      "question": "What is the purpose of model checking in Rhapsody?",
      "answer": "Model checking in Rhapsody is used to verify that a model satisfies specified properties, such as safety and liveness, ensuring that the system behaves as expected."
    },
    {
      "question": "How do I import an external model into Rhapsody?",
      "answer": "To import an external model into Rhapsody, use the 'Import' option from the File menu and select the appropriate file format, such as XMI, to bring the model into the Rhapsody environment."
    },
    {
      "question": "What is a deployment diagram in Rhapsody?",
      "answer": "A deployment diagram in Rhapsody is a type of UML diagram that shows the physical architecture of a system, including hardware components, nodes, and communication paths."
    },
    {
      "question": "How do I create a package diagram in Rhapsody?",
      "answer": "To create a package diagram in Rhapsody, select the 'Package Diagram' option and then group related classes, interfaces, or components into packages to organize your model."
    },
    {
      "question": "Can I create custom profiles in Rhapsody?",
      "answer": "Yes, you can create custom profiles in Rhapsody by defining new stereotypes, tagged values, and constraints tailored to specific domain requirements."
    },
    {
      "question": "How do I update a model in Rhapsody?",
      "answer": "To update a model in Rhapsody, open the model file, make the necessary changes using the graphical editor, and then save the updated model."
    },
    {
      "question": "How do I create a flowchart in Rhapsody?",
      "answer": "To create a flowchart in Rhapsody, use the 'Activity Diagram' option and define actions, decision nodes, and flows to represent the sequence of activities in a process."
    },
    {
      "question": "How can I back up my Rhapsody projects?",
      "answer": "To back up your Rhapsody projects, regularly save your work and create copies of your project files and related resources in a secure location, such as a version control repository."
    },
    {
      "question": "How do I configure a simulation environment in Rhapsody?",
      "answer": "To configure a simulation environment in Rhapsody, use the 'Simulation' settings to define parameters like the execution engine, logging options, and breakpoints."
    },
    {
      "question": "What is the purpose of the Object Model Diagram in Rhapsody?",
      "answer": "The Object Model Diagram in Rhapsody represents the structure of a system at a particular point in time, showing objects, their states, and their relationships."
    },
    {
      "question": "How do I link requirements to models in Rhapsody?",
      "answer": "To link requirements to models in Rhapsody, use the 'Requirements' view to associate requirements with specific model elements, ensuring traceability and validation."
    },
    {
      "question": "How do I run a script in Rhapsody?",
      "answer": "To run a script in Rhapsody, go to the 'Tools' menu, select 'Scripts', and choose the script you want to execute. You can write scripts in JavaScript or other supported languages."
    },
    {
      "question": "What is the use of the Rhapsody ReporterPLUS?",
      "answer": "Rhapsody ReporterPLUS is a tool that allows users to generate custom reports based on the models and data in Rhapsody, providing insights and documentation for various stakeholders."
    },
    {
      "question": "How do I perform reverse engineering in Rhapsody?",
      "answer": "To perform reverse engineering in Rhapsody, use the 'Reverse Engineering' feature to import existing code and generate corresponding models, helping you understand and visualize the system architecture."
    },
    {
      "question": "Which SysML diagram type is best suited for modeling system behaviors in response to external and internal events?",
      "answer": " State Machine Diagram."
    },
    {
      "question": "Customizing Rhapsody for specific project needs can lead to which of the following outcomes?",
      "answer": " Enhanced productivity through tailored modeling environments."
    },
    {
      "question": "In a state machine diagram, what does a ‘transition’ represent?",
      "answer": " A change in the state of an object in response to an event"
    },
    {
      "question": "Using Rhapsody's RELM integration can help teams achieve what outcome?",
      "answer": " Improve traceability and impact analysis for model changes."
    },
    {
      "question": "How can Rhapsody's integration with third-party analysis tools benefit a systems engineering project?",
      "answer": " Enhancing model validation and verification processes"

    },
    {
      "question": "When setting up a new project in Rhapsody, what is a critical step for ensuring model consistency across different team members?",
      "answer": " Defining a clear set of naming conventions."
    },
    {
      "question": "For modeling sequential interactions in a system, which SysML diagram provides the most clarity?",
      "answer": " Sequence Diagram."
    },
    {
      "question": "Which file format is commonly edited to customize profiles within Rhapsody?",
      "answer": " .xml."
    },
    {
      "question": "In IBM Rhapsody, which feature allows for the creation of custom profiles that can extend the modeling capabilities of the tool??",
      "answer": " The Profile Toolkit"
    },
    {
      "question": "What feature in Rhapsody allows for the visualization of the dynamic behavior of a system through simulations?",
      "answer": " Test Conductor."
    },
    {
      "question": "How can I optimize code generation settings in Rhapsody?",
      "answer": "Optimize code generation settings in Rhapsody by accessing the 'Code Generation' options under 'Tools', where you can adjust settings for optimization, memory usage, and target platform specifics."
    },
    {
      "question": "How do I handle errors in Rhapsody simulations?",
      "answer": "During simulations, errors can be handled by checking the error log, reviewing the model’s execution flow, and adjusting model elements to correct issues identified during the simulation."
    },
    {
      "question": "What is the purpose of the activity diagram in Rhapsody?",
      "answer": "Activity diagrams in Rhapsody are used to model the flow of control and data in a system, representing workflows, processes, and the sequence of actions."
    },
    {
      "question": "How can I enable debugging for generated code in Rhapsody?",
      "answer": "Enable debugging by setting breakpoints in the model, configuring the debug environment under 'Simulation' settings, and using the integrated debugger tools."
    },
    {
      "question": "What are the key benefits of using SysML in Rhapsody for systems engineering?",
      "answer": "SysML in Rhapsody provides enhanced capabilities for modeling requirements, system behaviors, structures, and interactions, improving traceability and design accuracy."
    },
    {
      "question": "How do I create custom code templates in Rhapsody?",
      "answer": "Custom code templates can be created in Rhapsody by accessing the 'Code Generation Templates' section, where you can define and modify templates to suit specific project needs."
    },
    {
      "question": "What is the purpose of the object-oriented analysis feature in Rhapsody?",
      "answer": "Object-oriented analysis in Rhapsody helps in identifying system components and their interactions early in the design phase, allowing for better system architecture design."
    },
    {
      "question": "Can Rhapsody models be exported to MATLAB/Simulink?",
      "answer": "Yes, Rhapsody supports exporting models to MATLAB/Simulink, enabling integrated simulation and analysis of control systems and real-time applications."
    },
    {
      "question": "How do I validate a state machine in Rhapsody?",
      "answer": "State machines can be validated in Rhapsody by running simulations, checking for logical consistency, and verifying that transitions and actions align with system requirements."
    },
    {
      "question": "What is the use of action language in Rhapsody?",
      "answer": "Action language in Rhapsody allows users to define detailed behaviors within models, such as state transitions, control logic, and variable manipulations using textual expressions."
    },
    {
      "question": "How does Rhapsody support Agile workflows?",
      "answer": "Rhapsody supports Agile workflows through iterative development, continuous integration, rapid prototyping, and flexible modeling, making it adaptable to frequent changes in requirements."
    },
    {
      "question": "How do I create a sequence of events in Rhapsody?",
      "answer": "Create a sequence of events in Rhapsody using sequence diagrams, where you can drag and drop events, messages, and objects to define the interaction flow."
    },
    {
      "question": "What is the purpose of the 'Requirement Traceability' feature in Rhapsody?",
      "answer": "Requirement Traceability in Rhapsody allows linking requirements to model elements, ensuring that all requirements are met and validated throughout the development process."
    },
    {
      "question": "How do I simulate time-based events in Rhapsody?",
      "answer": "Simulate time-based events in Rhapsody by defining timers and time triggers within state machines or activity diagrams, enabling real-time event handling in simulations."
    },
    {
      "question": "How can I generate documentation from Rhapsody models?",
      "answer": "Documentation can be generated from Rhapsody models using the built-in reporting tools, where you can export diagrams, code, and model details to formats like HTML, PDF, or Word."
    },
    {
      "question": "How do I use custom stereotypes in Rhapsody?",
      "answer": "Custom stereotypes in Rhapsody are used to extend UML elements with additional properties and can be defined in the Profile Editor to suit specific project requirements."
    },
    {
      "question": "What is model-based testing in Rhapsody?",
      "answer": "Model-based testing in Rhapsody involves creating test cases directly from system models, allowing for automated validation of system behaviors against expected outcomes."
    },
    {
      "question": "Can I import requirements from Excel into Rhapsody?",
      "answer": "Yes, Rhapsody supports importing requirements from Excel by using the import feature, mapping Excel columns to requirement attributes within Rhapsody."
    },
    {
      "question": "How do I create a swimlane in an activity diagram in Rhapsody?",
      "answer": "To create a swimlane in an activity diagram, use the 'Partition' option, which allows you to organize actions and control flows according to different roles or system components."
    },
    {
      "question": "How can I model hardware-software interaction in Rhapsody?",
      "answer": "Model hardware-software interaction using deployment diagrams to represent the physical connections between hardware components and software entities."
    },
    {
      "question": "What is the best way to organize large models in Rhapsody?",
      "answer": "Organize large models in Rhapsody using packages, which allow you to group related elements, creating a clear and manageable structure for complex systems."
    },
    {
      "question": "How do I simulate real-time constraints in Rhapsody?",
      "answer": "Real-time constraints can be simulated by defining time-based triggers, deadlines, and priority rules within state machines and verifying them using simulation tools."
    },
    {
      "question": "How can I secure my Rhapsody models?",
      "answer": "Secure your models by using version control, setting up user access permissions, and regularly backing up your project files to prevent data loss and unauthorized changes."
    },
    {
      "question": "What is a communication diagram in Rhapsody?",
      "answer": "A communication diagram in Rhapsody shows the interactions between objects and components in a system, focusing on the structural relationships and message flow."
    },
    {
      "question": "How do I set breakpoints in Rhapsody simulations?",
      "answer": "Set breakpoints by clicking on model elements or lines of code within the simulation environment, allowing you to pause execution and examine system states."
    },
    {
      "question": "How can Rhapsody help in compliance with industry standards?",
      "answer": "Rhapsody helps in compliance by providing modeling templates, validation rules, and code generation settings aligned with industry standards like ISO 26262 and DO-178C."
    },
    {
      "question": "What is the purpose of the class diagram in Rhapsody?",
      "answer": "Class diagrams in Rhapsody are used to model the static structure of a system, showing classes, attributes, methods, and the relationships between them."
    },
    {
      "question": "How do I link Rhapsody to external databases?",
      "answer": "Link Rhapsody to external databases by using APIs or plugins that support data synchronization and integration, facilitating seamless data flow between the model and external sources."
    },
    {
      "question": "Can Rhapsody models be integrated with cloud services?",
      "answer": "Yes, Rhapsody models can be integrated with cloud services using APIs and connectors, enabling remote collaboration, model sharing, and data analysis."
    },
    {
      "question": "How do I visualize data flow in Rhapsody?",
      "answer": "Visualize data flow using data flow diagrams, which represent how data moves between components, processes, and storage within the system."
    },
    {
      "question": "What are the key considerations when setting up Rhapsody for a new project?",
      "answer": "Key considerations include defining modeling standards, setting up version control, customizing code generation templates, and configuring simulation environments."
    },
    {
      "question": "How do I model complex algorithms in Rhapsody?",
      "answer": "Model complex algorithms using state machines, activity diagrams, and sequence diagrams to capture the logic, control flow, and data processing steps."
    },
    {
      "question": "What is a context diagram in Rhapsody?",
      "answer": "A context diagram in Rhapsody provides a high-level view of a system, showing its boundaries, external entities, and major interactions."
    },
    {
      "question": "How do I configure user roles in Rhapsody?",
      "answer": "Configure user roles by accessing the 'Project Settings', where you can assign permissions and define user access levels for different parts of the project."
    },
    {
      "question": "What are the benefits of using templates in Rhapsody?",
      "answer": "Templates in Rhapsody streamline model creation by providing predefined structures, ensuring consistency, and reducing setup time for new projects."
    },
    {
      "question": "How can I track model changes in Rhapsody?",
      "answer": "Track model changes using version control integration, change logs, and Rhapsody's built-in change tracking features to monitor modifications and updates."
    },
    {
      "question": "How do I create a communication path in Rhapsody?",
      "answer": "Create a communication path in Rhapsody by linking elements in communication or sequence diagrams, defining the flow of messages between objects."
    },
    {
      "question": "What is the use of the properties view in Rhapsody?",
      "answer": "The properties view provides detailed information about selected model elements, allowing users to adjust settings, attributes, and behaviors directly."
    },
    {
      "question": "Can Rhapsody generate code for multiple programming languages?",
      "answer": "Yes, Rhapsody supports code generation for multiple languages, including C, C++, Java, and Ada, making it versatile for various software development needs."
    },
    {
      "question": "How do I perform model validation in Rhapsody?",
      "answer": "Model validation in Rhapsody can be performed using built-in validation tools that check for compliance with modeling rules, syntax correctness, and logical consistency."
    },
    {
      "question": "What is the role of the model browser in Rhapsody?",
      "answer": "The model browser in Rhapsody helps navigate through the model's hierarchy, providing a structured view of all elements, diagrams, and relationships."
    },
    {
      "question": "How can I define custom attributes in Rhapsody?",
      "answer": "Custom attributes can be defined in Rhapsody by editing the properties of elements, allowing you to add specific data fields to tailor the model to your needs."
    },
    {
      "question": "How do I set up notifications for changes in Rhapsody?",
      "answer": "Set up notifications by configuring project settings or integrating with a version control system that can alert you to changes in the model or codebase."
    },
    {
      "question": "What are interaction diagrams used for in Rhapsody?",
      "answer": "Interaction diagrams in Rhapsody are used to detail how objects interact within a system, focusing on message passing, sequences, and collaboration among components."
    },
    {
      "question": "How can I create a reusable component in Rhapsody?",
      "answer": "Create reusable components by defining classes, blocks, or packages that encapsulate functionality, allowing them to be reused across different models and projects."
    },
    {
      "question": "How do I measure model performance in Rhapsody?",
      "answer": "Measure model performance using Rhapsody's profiling and analysis tools, which evaluate the efficiency of simulations, code generation, and execution speed."
    }
  ]
}


# Load the dataset into a DataFrame
df = pd.DataFrame(data["data"])

# Create a context from your dataset
context = " ".join(df["answer"])

# Vectorize the questions for similarity check
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["question"])

# Function to find the most similar question above a similarity threshold
def get_most_similar_answer(user_question, threshold=0.5):
    user_question_vec = vectorizer.transform([user_question])
    similarities = cosine_similarity(user_question_vec, X).flatten()
    # Find the index of the most similar question
    best_match_index = similarities.argmax()
    best_similarity = similarities[best_match_index]
    # Return the answer if the similarity is above the threshold
    if best_similarity >= threshold:
        return df["answer"].iloc[best_match_index], df["question"].iloc[best_match_index], best_similarity
    return None, None, best_similarity

# Function to find similar questions to the user's input
def find_similar_questions(user_question, top_n=3):
    user_question_vec = vectorizer.transform([user_question])
    similarities = cosine_similarity(user_question_vec, X).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]
    return [(df["question"].iloc[i], df["answer"].iloc[i]) for i in top_indices if similarities[i] > 0]

# Function to answer questions using the pretrained model
def answer_question(question, context):
    try:
        result = qa_pipeline({"question": question, "context": context})
        return result['answer']
    except Exception as e:
        return "I'm not sure how to answer that."

# Streamlit App Interface
st.title('Rhapsody Chatbot')
user_input = st.text_input("Ask a question about Rhapsody:")

# Handle different types of user input
if user_input:
    if user_input.lower() in ["hi", "hello", "hey"]:
        st.write("Hello! How can I help you with Rhapsody today?")
    else:
        # Check for a similar question above the threshold
        dataset_answer, matched_question, similarity = get_most_similar_answer(user_input, threshold=0.5)
        
        if dataset_answer:
        
            response = dataset_answer
        else:
            response = "Ask question related to Rhapsody"
        st.write(f"Answer: {response}")

        # Asking if the user needs suggestion questions
        suggest = st.radio("Do you need any suggestion questions?", ("No", "Yes"))

        if suggest == "Yes":
            similar_questions = find_similar_questions(user_input)
            question_choices = [q for q, _ in similar_questions]
            if question_choices:
                selected_question = st.selectbox("Select a question:", question_choices)

                if selected_question:
                    # Find the answer corresponding to the selected question
                    answer = dict(similar_questions)[selected_question]
                    st.write(f"Answer: {answer}")
            else:
                st.write("No similar questions found.")

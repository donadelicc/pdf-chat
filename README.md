PDF-CHAT
About the Project
PDF-CHAT is a web application developed using Streamlit, enabling users to upload a PDF file and interact with its content through a chat interface. The application extracts text from the PDF file, performs semantic search, and uses a language model to generate responses based on the document's text.

Features
PDF file upload and text extraction.
Semantic search within the PDF text.
Question and answer functionality with an AI model.
Technologies
Streamlit
PyPDF2
Langchain
FAISS
OpenAI
Installation
Before you begin, make sure you have Python installed on your system.

Clone this repository:
bash
Copy code
git clone [repo-url]
Install dependencies:
Copy code
pip install -r requirements.txt
Setup
To run PDF-CHAT:

Download the .env file (if necessary) and place it in the project root folder.
Run the Streamlit application:
arduino
Copy code
streamlit run app.py
Usage
After starting the application:

Upload a PDF file through the user interface.
Ask questions related to the content of the PDF file in the text box.
View the responses generated by the AI model.
Contributions
Contributions to this project are welcome. Please follow these steps to contribute:

Fork the repository.
Create a new branch with your feature or fix.
Submit a pull request for review.
License
Specify the license you are using for your project (e.g., MIT, GPL, etc.).
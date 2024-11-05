from django.shortcuts import render
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_google_genai import GoogleGenerativeAI

def upload_csv(request):
    # Retrieve the DataFrame and file name from the session, if they exist
    data = pd.read_json(request.session['data']) if 'data' in request.session else None
    file_name = request.session.get('file_name', None)
    additional_info = None
    reqd_response = None

    if request.method == "POST":
        # Get the uploaded file and additional info from the form
        csv_file = request.FILES.get('csv_file')
        additional_info = request.POST.get('text_input')

        if csv_file:
            # Read the CSV file into a DataFrame
            data = pd.read_csv(csv_file)

            # Store the DataFrame and file name in the session
            request.session['data'] = data.to_json()
            request.session['file_name'] = csv_file.name

        if data is not None and additional_info:
            # Initialize the LLM here (ensure that the Google API key is configured)
            import os
            os.environ["GOOGLE_API_KEY"] = "<GOOGLE-GEMINI-API-KEY>"
            llm = GoogleGenerativeAI(model="gemini-pro")

            # Create a Pandas DataFrame agent
            agent = create_pandas_dataframe_agent(llm, data, verbose=True, handle_parsing_errors=True)
            response = agent.invoke(additional_info)
            reqd_response = response['output']

    # Pass the file name, data, and response to the template
    return render(request, 'myapp/index.html', {
        'data': data, 
        'file_name': file_name, 
        'additional_info': reqd_response
    })

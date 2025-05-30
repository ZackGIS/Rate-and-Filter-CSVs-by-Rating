import os
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import time
from multiprocessing import Pool


os.environ["OPENAI_API_KEY"] = "Your API key goes here"

print(os.getenv("OPENAI_API_KEY")) #Make sure to right key is used

inputFolderPath = r'C:\InternCSVs\GIS_Web_Services\output2\Batch3'
outputFolderPath = r'C:\InternCSVs\GIS_Web_Services\output\Runthrough3'
os.makedirs(outputFolderPath, exist_ok=True)

tagging_prompt = ChatPromptTemplate.from_template(
    """
    You are an expert in energy data classification. 
    Your job is to extract and classify the relevant properties related to energy from the passage below.

    The information you need to extract is:
    1. Energy aspect: Describe the specific energy-related aspect (e.g., oil, gas, renewables).
    2. Energy relevance: On a scale of 1 to 10, rate how closely related the input is to the energy aspect you provided.
    3. Tag: Choose a relevant tag from the following list based on the context: 
       ["wells", "pipelines", "infrastructure", "imagery", "weather", "environmental", "geology", 
       "seismic", "geomatics", "renewables", "emissions", "basemaps", "bathymetry", "licenses", 
       "blocks", "leases", "oil", "gas", "topography", "gravity", "magnetics"].

    Only extract these three pieces of information and format the output according to the Classification(BaseModel).
    Now, classify the following text from the data source:
    {input}
    """
)


class Classification(BaseModel):
    energy: str = Field(..., description="What aspect of energy is this related to?")
    energy_related: int = Field(..., description="How related is this to energy on a scale from 1 to 10?")
    tag: str = Field(
        ..., description="Please select a tag from the provided list.",
        enum=[
            "wells", "pipelines", "infrastructure", "imagery", "weather",
            "environmental", "geology", "seismic", "geomatics", "renewables",
            "emissions", "basemaps", "bathymetry", "licenses", "blocks", "leases",
            "oil", "gas", "topography", "gravity", "magnetics"
        ]
    )

# processFile uses a gpt-3.5-turbo llm from OpenAI (there are better models) to tag/classify data from the dataframe
# with the 3 label fields defined in the Classification basemodel
def processFile(filename):
    print(f"Starting processing for file: {filename}")
    llm = ChatOpenAI(model="gpt-4-turbo", temperature=0, max_tokens=None, timeout=None).with_structured_output(Classification)
    tagging_chain = tagging_prompt | llm #tagging chain uses combines the tagging prompt (which takes in input description_text, tags_text, and title_text defined below) with the
                                        # llm.

    csvFilePath = os.path.join(inputFolderPath, filename)
    data = pd.read_csv(csvFilePath) #Using Pandas df to declare the dataframe object

    #Creating the extra columns in the dataframe to hold the output from the LLM
    data['energy'] = None
    data['energy_related'] = None
    data['tag'] = None

    for index, row in data.iterrows():  # loop through the entire dataframe.
        description_text = row['description'] # Description text is description row header
        tags_text = row['tags']  # Same with tags
        title_text = row['title'] # Same with title

        # Some dataframe entries have a "No description" so we take in other input text in that case. Also
        # could read input from every header index.
        text = description_text and title_text if description_text != "No description" else tags_text and title_text
        prompt_input = {"input": text} # declare the input text to be passed to the tagging_chain

        try:
            print(f"Making API call for row {index} in {filename}")
            result = tagging_chain.invoke(prompt_input) # Invoke function is called on the taggin_chain object
            data.at[index, 'energy'] = result.energy
            data.at[index, 'energy_related'] = result.energy_related
            data.at[index, 'tag'] = result.tag
            time.sleep(2.65)
        except Exception as e:
            print(f"Error {index} in {filename}: {e}")
            continue

    #filter the data based on 'energy_related' score between and including 7-10
    filtered_data = data[(data['energy_related'] >= 7) & (data['energy_related'] <= 10)]


    outputFilePath = os.path.join(outputFolderPath, filename[:-4] + "_filtered.csv")
    print(f"Saving filtered data for file: {filename}")
    filtered_data.to_csv(outputFilePath, index=False)


def main():
    csv_files = []
    for filename in os.listdir(inputFolderPath):
        if filename.endswith('.csv'):
            csv_files.append(filename)

    with Pool(processes=4) as pool:
        pool.map(processFile, csv_files)


if __name__ == "__main__":
    main()


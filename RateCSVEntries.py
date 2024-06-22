from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import pandas as pd
import openai
import os

# API key
os.environ["OPENAI_API_KEY"] = "NA- it's saved in a separate file"
openai.api_key = os.getenv("OPENAI_API_KEY")
#print(f"API Key: {openai.api_key}")

# input and output folder paths
inputFolderPath = r'C:\InternCSVs\GIS_Web_Services'
outputFolderPath = r'C:\InternCSVs\GIS_Web_Services\output'
os.makedirs(outputFolderPath, exist_ok=True)

# telling the llm to look at the Classification for instructions.
tagging_prompt = ChatPromptTemplate.from_template(
    """
Extract the desired information from the following passage.

Only extract the properties mentioned in the 'Classification' function.

Passage:
{input}
"""
)


# Borrowed from the example code. This class defines the structure of the llm output.
class Classification(BaseModel):


    energy: str = Field(description="What aspect of energy is this related to?")

    # this is the rating determining how related to energy each entry in the csv is. 1-very low, 10-very high.
    energy_related: int = Field(description="How related is this to energy from 1 to 10")

    # values in the enum list represent different categories the llm will classify entries into after giving
    # an "energy related rating" from 1-10.
    tag: str = Field(
        ..., enum=[
            "wells", "pipelines", "infrastructure", "imagery", "weather",
            "environmental", "geology", "seismic", "geomatics", "renewables",
            "emissions", "basemaps", "bathymetry"
        ]
    )


llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, max_tokens=None,
                 timeout=None).with_structured_output(Classification)

tagging_chain = tagging_prompt | llm


# tagging_function from the code example provided.
def tagging_function(text):
    result = tagging_chain.invoke({"input": text})
    return {
        'energy': result.energy,
        'energy_related': result.energy_related,
        'tag': result.tag
    }


for filename in os.listdir(inputFolderPath):     # loop through each file in input folder
    if filename.endswith('.csv'):    # make sure it's a csv file first
        csvFilePath = os.path.join(inputFolderPath, filename)    # specify filepath for each csv in the folder
        data = pd.read_csv(csvFilePath)      # creating a pandas dataframe object

        # these lines are initializing new columns in the csv files i think.
        # they won't be saved into the output csv. Just used for filtering the original files
        data['energy'] = None
        data['energy_related'] = None
        data['tag'] = None

        # replacing any '_' in the title column with a space. Needed so the llm can process the title text.
        data['title_str'] = data['title'].astype(str).str.replace('_', ' ')

        # not certain we need this but including it anyway. It's adding a title length column to the csv
        # but it won't be saved in the output like the others above.
        data['title_len'] = data['title'].apply(lambda x: len(str(x).split()))

        # start looping over the data frame.
        for index, row in data.iterrows():

            # news_text stores the processed title text. got this from the example code.
            news_text = row['title_str']

            # calling the tagging_function with parameter news_text.
            tagging_result = tagging_function(news_text)

            # updating energy, energy_related, and tag at the current index, row of the DF with the results retruned by
            # the tagging_function
            data.at[index, 'energy'] = tagging_result['energy']
            data.at[index, 'energy_related'] = tagging_result['energy_related']
            data.at[index, 'tag'] = tagging_result['tag']

        # now choosing only results with a "high" energy-related rating to be the output for the new CSVs.
        filtered_data = data[(data['energy_related'] >= 7) & (data['energy_related'] <= 10)]

        outputFilePath = os.path.join(outputFolderPath, filename[:-4] + "_filtered.csv")

        # save the new CSVs but don't include any of the temporary columns generated above. I don't think we want
        # the format changed. Just the most relevant feature layers.
        filtered_data.to_csv(outputFilePath, index=False, columns=['title', 'url', 'type', 'tags', 'description',
                                                                   'thumbnailurl'])

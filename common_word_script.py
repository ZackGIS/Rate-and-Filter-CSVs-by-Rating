import pandas as pd
import re
from collections import Counter
from nltk.corpus import stopwords

# Ensure you have the stopwords from NLTK
import nltk
nltk.download('stopwords')

# Sample data frame for demonstration
data = {
    'title': [
        'This_is_a_testTitle',
        'Another_example_ofTitle',
        'Fish_and_Chips'
    ]
}
df = pd.DataFrame(data)

# Function to process titles
def process_title(title):
    # Replace underscores with spaces
    title = title.replace('_', ' ')
    # Add spaces between lowercase and uppercase letters
    title = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', title)
    return title

# Apply the function to the title column
df['processed_title'] = df['title'].apply(process_title)

# Function to count words, excluding stop words
def count_words(titles):
    stop_words = set(stopwords.words('english'))
    word_count = Counter()
    
    for title in titles:
        words = title.split()
        for word in words:
            word_lower = word.lower()
            if word_lower not in stop_words:
                word_count[word_lower] += 1
                
    return word_count

# Count words in the processed titles
word_counts = count_words(df['processed_title'])

# Display the processed titles and word counts
print("Processed Titles:")
print(df['processed_title'])
print("\nWord Counts:")
print(word_counts)

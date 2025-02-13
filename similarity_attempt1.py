# -------------------------------------------------------------------------
# AUTHOR: Gabriel Alfredo Siguenza
# FILENAME: similarity.py
# SPECIFICATION: python program that finds 2 documents with cosine similarities in csv file
# FOR: CS 5990 (Advanced Data Mining) - Assignment #1
# TIME SPENT: approx 6.5 hours total, 1.5 for python program
# -----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy,
#pandas, or other sklearn modules.
#You have to work here only with standard dictionaries, lists, and arrays

# Importing some Python libraries
import csv
from sklearn.metrics.pairwise import cosine_similarity

documents = []

#reading the documents in a csv file
with open('cleaned_documents.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    header = next(reader)  # Skip the header row
    for row in reader:
        documents.append(row[1])  # assuming each document text is in the second column

# Printing documents to verify their content
# print("\nDocuments:", documents)

#Building the document-term matrix by using binary encoding.
#You must identify each distinct word in the collection without applying any transformations, using
# the spaces as your character delimiter.
#--> add your Python code here
unique_words = set()
for document in documents:
    words = document.split()
    unique_words.update(words)
    print(f"Words in Document: {words}")

unique_words = list(unique_words)

# Printing distinct words to verify
# print("\nDistinct Words:", distinct_words)

# Creating the binary encoded document-term matrix
docTermMatrix = []
for document in documents:
    word_vector = [1 if word in document.split() else 0 for word in unique_words]
    docTermMatrix.append(word_vector)

# Printing docTermMatrix to verify its content
print("\nDocument-Term Matrix:")
for vector in docTermMatrix:
    print(vector)

# Compare the pairwise cosine similarities and store the highest one
# Use cosine_similarity([X], [Y]) to calculate the similarities between 2 vectors
# --> Add your Python code here
similarity_matrix = cosine_similarity(docTermMatrix)
max_similarity = 0
similar_docs = (None, None)

for i in range(len(similarity_matrix)):
    for j in range(i + 1, len(similarity_matrix)):
        similarity = similarity_matrix[i][j]
        if similarity > max_similarity:
            max_similarity = similarity
            similar_docs = (i, j)

# Print the highest cosine similarity following the information below
# The most similar documents are document 10 and document 100 with cosine similarity = x
# --> Add your Python code here
print(f"\nThe most similar documents are {similar_docs[0]} and {similar_docs[1]} with a similarity of {max_similarity}")
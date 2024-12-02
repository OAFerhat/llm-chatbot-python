import streamlit as st
from llm import llm
from graph import graph
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from langchain.prompts.prompt import PromptTemplate

CYPHER_GENERATION_TEMPLATE = """
You are an expert Neo4j Developer translating user questions into Cypher to answer questions about movies and provide recommendations.
Convert the user's question based on the schema.

Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.

Do not return entire nodes or embedding properties.

Fine Tuning:

For movie titles that begin with "The", move "the" to the end. For example "The 39 Steps" becomes "39 Steps, The" or "the matrix" becomes "Matrix, The".


Example Cypher Statements:

1. To find who acted in a movie:
```
MATCH (p:Person)-[r:ACTED_IN]->(m:Movie {{title: "Movie Title"}})
RETURN p.name, r.role
```

2. To find who directed a movie:
```
MATCH (p:Person)-[r:DIRECTED]->(m:Movie {{title: "Movie Title"}})
RETURN p.name
```

3. To find the degrees of separation between two people:
```
MATCH p=shortestPath((actor1:Actor)-[*]-(actor2:Actor))
WHERE actor1.name = 'Actor 1' AND actor2.name = 'Actor 2'
RETURN length(p) - 1 as degrees_of_separation
```
Schema:
{schema}

Question:
{question}

Cypher Query:
"""

cypher_prompt = PromptTemplate.from_template(CYPHER_GENERATION_TEMPLATE)


def format_cypher_response(result):
    if not result:
        return "I don't have any information about that."
    
    # Handle degrees of separation query
    if 'degrees_of_separation' in result[0]:
        degrees = result[0]['degrees_of_separation']
        return f"The degrees of separation is {degrees}."
    
    return str(result)

# Create the Cypher QA chain
cypher_qa = GraphCypherQAChain.from_llm(
    llm,
    graph=graph,
    verbose=True,
    allow_dangerous_requests=True,  # Only enable this if you understand the security implications
    cypher_prompt=cypher_prompt,
    return_direct=True,
    return_intermediate_steps=True,
    post_process=format_cypher_response
)

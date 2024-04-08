# CS6804 Final Project

## Trustworthiness of LLM Feedback on Communication Utterances

This final project attempts to understand how human judges and LLM judges perceive the trustworthiness of LLM generated feedback on human communication utterances. Feedback is all generated by an LLM coach agent who has been prompted to be an expert leadership coach. Feedback either has no retrieval augmented generation (RAG) labeled as "none", local RAG from PDFs, or web search RAG from Google search results.

## Data

The data is stored in the `data` directory. The feedback which is used to generate the RAG options is provided in `feedback_msgs.json`. Results from the LLM judges are given in their respective JSON files. Results from the human judge battles is found at the HuggingFace dataset `lancewilhelm/cs6804_final`. 

### Knowledge

The local knoweledge used for the local RAG method can be found in `knowledge/`.

### Local RAG Vector DB

The results from document loading and embedding are stored in `storage`. This is an sklearn vector database. 

## Code

`final_project.ipynb` contains the code used to generate the local and web RAG versions of each feedback item. It also contains the code used to generate the LLM judge results.

`battle_app/` contains the code used to generate the HuggingFace space that was provided to human judges for feedback evaluation. This space can be found [here](https://huggingface.co/spaces/lancewilhelm/cs6804_final_battle).

`analysis.ipynb` contains all of the EDA of the results and the power analysis used to understand how many samples were needed. This file also contains the hypothesis testing used to determine if the different RAG methods significantly affected the perceived trustworthiness.
from langchain.load import dumps, loads
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever,BM25Retriever, EnsembleRetriever

def query_prompt(query, count):
    prompt = f'''- Given a single input query generate only {count} search queries based on the input query.
- The queries should be in context with the input query while maintaining the orignal intent.
- Output should be in the form of a python list  no other format.
- Do not generate any extra irrelevant text other than the list.

Use the example format below to understand how to generate your output.
- Examples

    "question": "What is the best way to learn a new programming language?",
    "rephrased_questions": [
      "How can I effectively learn a new programming language?",
      "What strategies are most effective for mastering a new programming language?",
      "What are the most efficient methods for acquiring proficiency in a new programming language?",
      "What approaches can I take to successfully learn and utilize a new programming language?"
    ]

    "question": "What are some tips for writing a compelling essay?",
    "rephrased_questions": [
      "How can I craft an engaging and impactful essay?",
      "What techniques can be employed to enhance the effectiveness of an essay?",
      "What are some strategies to create a well-written and persuasive essay?",
      "What guidelines can be followed to produce a captivating and memorable essay?"
    ]

    "question": "What are the causes of climate change?",
    "rephrased_questions": [
      "What factors contribute to the phenomenon of climate change?",
      "What are the underlying mechanisms driving the Earth's climate transformation?",
      "What activities and natural processes are responsible for the changing climate?",
      "What are the actions and processes that leads to climate change?"
    ]
    
"question":{query}\n'''
    return prompt
    

def result_prompt(query, text):
    prompt = f'''{text}
    -Answer the question by extracting Extract key information from the text provided.
    -Keep it concise the answer should look like a summary.
    -Use the information provided in the text and dont answer from anywhre else.s
    
"question:{query}\n'''
    return prompt


def reciprocal_rank_fusion(results: list[list], k=60):
    fused_scores = {}
    for docs in results:
        # Assumes the docs are returned in sorted order of relevance
        for rank, doc in enumerate(docs):
            print(doc)
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            previous_score = fused_scores[doc_str]
            fused_scores[doc_str] += 1 / (rank + k)

    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return reranked_results


def reciprocal_rank_fusion_weighted(results: list[list], k=60, weight_factor=1.5):
    """
    Fuses results from multiple retrieval models using weighted reciprocal rank score.

    Args:
        results: List of lists, where each inner list represents the search results from a different retrieval model.
        k: Parameter controlling importance of results further down the list.
        weight_factor: Factor to apply to the scores of the first set of results.

    Returns:
        List of reranked documents and their corresponding scores.
    """

    fused_scores = {}
    for i, docs in enumerate(results):
        # Assumes the docs are returned in sorted order of relevance
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # Apply weight factor to the first set of results
            weight = weight_factor if i == 0 else 1
            fused_scores[doc_str] += weight * (1 / (rank + k))

    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return reranked_results


def fusion_formator(retriever,multi_query):

    mulri_results = []
    question_wise = []

    for query in multi_query:


    # Retrieve relevant documents (limited to 3)
        result = retriever.get_relevant_documents(query)
        
        for text in result:
            # Populate an empty dictionary for each document
            empty = {}
            empty['title'] = query
            empty['text'] = text
            
            # Append the dictionary to the question-wise results
            question_wise.append(empty)
        
    # Add the question-wise results for this query to the overall list
    mulri_results.append(question_wise)
    return mulri_results


def fusion_formator_compressed(langchain_model,retriever,multi_query):

    compressor = LLMChainExtractor.from_llm(langchain_model)
    compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)

    mulri_results = []
    question_wise = []

    for query in multi_query:


    # Retrieve relevant documents (limited to 3)
        compressed_docs = compression_retriever.get_relevant_documents(query)
        
        for text in compressed_docs:
            # Populate an empty dictionary for each document
            empty = {}
            empty['title'] = query
            empty['text'] = text
            
            # Append the dictionary to the question-wise results
            question_wise.append(empty)
        
    # Add the question-wise results for this query to the overall list
    mulri_results.append(question_wise)
    return mulri_results
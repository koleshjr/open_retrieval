class Config:
    rephrasing_prompt = """
    Please rephrase the following query into four different but related queries.
    Ensure each query is distinct in terms of phrasing, focus, or specificity.
    Act: {page_text}
    output : Format all responses as JSON objects as shown in the examples above. Should be in this format\n{format_instructions}
    """
    
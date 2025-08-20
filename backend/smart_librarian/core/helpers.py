from openai import OpenAI


def is_safe(text: str, client: OpenAI) -> bool:
    """
    Checks if a given text is safe using the OpenAI Moderation API.

    Args:
        text: The input text to moderate.
        client: An active OpenAI client instance.

    Returns:
        True if the text is deemed safe, False if it is flagged in any category.
    """
    try:
        response = client.moderations.create(model="text-moderation-latest", input=text)

        # The result object contains a list of results, one for each input string.
        # Since we only send one string, we check the first result.
        result = response.results[0]

        # 'flagged' is a boolean that is True if the content violates OpenAI's
        # usage policies in any category.
        return not result.flagged

    except Exception as e:
        # If the moderation API call fails for any reason, it's safer to
        # assume the content might be unsafe. We log the error and return False.
        print(f"Error during moderation API call: {e}")
        return False

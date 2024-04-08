import os
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())  # read local .env file
api_key = os.environ["OPENAI_API_KEY"]
if os.environ.get('OPENAI_ORG') is not None:
    organization = os.environ["OPENAI_ORG"]
    client = OpenAI(organization=organization, api_key=api_key)
else:
    client = OpenAI(api_key=api_key)

# Setup the LLM Agent class without langchain
class OpenAIChatLLMAgent:
    """This class is a wrapper for the OpenAI Chat API. It is designed to be used as a chatbot agent."""

    def __init__(
        self,
        system_prompt: str = '',
        model: str = "gpt-4-0125-preview",
        temperature: float = 0.0,
        output_format: str = None,
        store_responses: bool = True,
        response_format: dict = {'type':'text'}
    ):
        """Initialize the OpenAI Chat API wrapper.

        Args:
            system_prompt (str, optional): initial system prompt to the LLM. Defaults to "You are a helpful assistant.".
            model (str, optional): one of the OpenAI chat models. Defaults to "gpt-3.5-turbo".
            temperature (float, optional): what sampling temperature to use, between 0 and 2. Defaults to 0.0.
            output_format (str, optional): what output format to use. Defaults to None.
            store_responses (bool, optional): whether to store responses in the messages. Defaults to True.
        """
        self.model = model
        self.messages = []
        self.system_prompt = system_prompt
        self.output_format = output_format
        self.temperature = temperature
        self.store_responses = store_responses
        self.response_format = response_format

    # Getters and setters
    @property
    def messages(self):
        """Get the messages that have been sent to the chatbot.

        Returns:
            list: list of messages sent to the chatbot
        """
        return self._messages

    @messages.setter
    def messages(self, value: list):
        """Set the messages that have been sent to the chatbot.

        Args:
            value (list): list of messages sent to the chatbot
        """
        self._messages = value

    @property
    def model(self):
        """Get the OpenAI chat model that is being used.

        Returns:
            str: OpenAI chat model
        """
        return self._model

    @model.setter
    def model(self, value: str):
        """Set the OpenAI chat model that is being used.

        Args:
            value (str): OpenAI chat model
        """
        self._model = value

    @property
    def temperature(self):
        """Get the sampling temperature that is being used.

        Returns:
            float: sampling temperature
        """
        return self._temperature

    @temperature.setter
    def temperature(self, value: float):
        """Set the sampling temperature that is being used.

        Args:
            value (float): sampling temperature
        """
        self._temperature = value

    @property
    def output_format(self):
        """Get the output format that is being used.

        Returns:
            str: output format
        """
        return self._output_format

    @output_format.setter
    def output_format(self, value: str):
        """Set the output format that is being used.

        Args:
            value (str): output format
        """
        self._output_format = value

    @property
    def store_responses(self):
        """Get whether responses are being stored.

        Returns:
            bool: whether responses are being stored
        """
        return self._store_responses

    @store_responses.setter
    def store_responses(self, value: bool):
        """Set whether responses are being stored.

        Args:
            value (bool): whether responses are being stored
        """
        self._store_responses = value

    @property
    def system_prompt(self):
        """Get the system prompt that is being used.

        Returns:
            str: system prompt
        """
        return self._system_prompt
    
    @system_prompt.setter
    def system_prompt(self, value: str):
        """Set the system prompt that is being used.

        Args:
            value (str): system prompt
        """
        self._system_prompt = value
        if self.messages:
            self.messages[0]['content'] = value
        else:
            self.messages.append({"role": "system", "content": value})

    @property
    def response_format(self):
        """Get the response format that is being used.

        Returns:
            dict: response format
        """
        return self._response_format
    
    @response_format.setter
    def response_format(self, value: dict):
        """Set the response format that is being used.

        Args:
            value (dict): response format
        """
        self._response_format = value

    # Methods
    def add_message(self, role: str, content: str):
        """Add a message to the chatbot.

        Args:
            role (str): role of the message, either "user" or "system", or "assistant" if you want to inject a message from the assistant
            content (str): content of the message
        """
        if self.output_format is not None:
            content += f"\n\nOutput format:\n{self.output_format}"
        self.messages.append({"role": role, "content": content})

    def change_message(self, index: int, role: str, content: str):
        """Change a message in the chatbot.

        Args:
            index (int): index of the message to change
            role (str): role of the message, either "user" or "system", or "assistant" if you want to inject a message from the assistant
            content (str): content of the message
        """
        self.messages[index] = {"role": role, "content": content}

    def delete_message(self, index: int):
        """Delete a message from the chatbot.

        Args:
            index (int): index of the message to delete
        """
        del self.messages[index]

    def clear_memory(self):
        """Clear all messages from the chatbot except the system prompt."""
        self.messages = []
        if self.system_prompt is not None:
            self.messages.append({"role": "system", "content": self.system_prompt})

    def run(self, message: str = None, temporary_message: bool = False):
        """Run the chatbot with a message.

        Args:
            message (str, optional): message to send to the chatbot. Defaults to None, uses current messages if None.
        Returns:
            str: response from the chatbot
        """
        # Add a message if one was provided
        if message != None:
            self.add_message("user", message)
        
        # Get the response from the API
        response = client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            temperature=self.temperature,
            stream=False,
            response_format=self.response_format
        )

        # Delete the temporary message if it was added
        if temporary_message:
            self.delete_message(-1)

        # Store the response if requested
        if self.store_responses:
            self.add_message(
                response.choices[0].message.role,
                response.choices[0].message.content,
            )

        return response

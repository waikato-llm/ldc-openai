import argparse
import copy
import tiktoken
from typing import List, Optional

from ldc.core import LOGGING_WARN, DOMAIN_PAIRS, DOMAIN_PRETRAIN, DOMAIN_TRANSLATION
from ldc.core import LOCATION_ANY, LOCATION_INSTRUCTION, LOCATION_INPUT, LOCATION_OUTPUT, LOCATION_CONTENT, \
    LOCATIONS, LOCATIONS_PAIRS, LOCATIONS_PRETRAIN
from ldc.filter import Filter
from ldc.pretrain import PretrainData
from ldc.supervised.pairs import PairData
from ldc.translation import TranslationData


class OpenAICountTokens(Filter):
    """
    Counts text tokens for OpenAI.
    """

    def __init__(self, encoding: str = None, model: str = None, prompt: str = None, price_per_1k_tokens: float = None,
                 location: str = LOCATION_ANY, languages: List[str] = None,
                 logger_name: str = None, logging_level: str = LOGGING_WARN):
        """
        Initializes the filter. Either encoding or model need to be provided.

        :param encoding: the encoding to use, eg cl100k_base, p50k_base, r50k_base, ...
        :type encoding: str
        :param model: the model to get the encoding for, eg gpt-4, gpt-3.5-turbo, text-davinci-002, ...
        :type model: str
        :param prompt: the prompt to use for each query (addings its # of tokens to the total with each string)
        :type prompt: str
        :param price_per_1k_tokens: the price for 1000 tokens
        :type price_per_1k_tokens: float
        :param location: which part of the data to count the tokens
        :type location: str
        :param languages: the languages to restrict the check to, None to check all
        :type languages: list
        :param logger_name: the name to use for the logger
        :type logger_name: str
        :param logging_level: the logging level to use
        :type logging_level: str
        """
        super().__init__(logger_name=logger_name, logging_level=logging_level)

        if location not in LOCATIONS:
            raise Exception("Invalid location: %s" % location)

        self.encoding = encoding
        self.model = model
        self.prompt = prompt
        self.price_per_1k_tokens = price_per_1k_tokens
        self.location = location
        self.languages = languages
        self._count = 0
        self._encoding = None
        self._count_prompt = 0

    def name(self) -> str:
        """
        Returns the name of the handler, used as sub-command.

        :return: the name
        :rtype: str
        """
        return "openai-count-tokens"

    def description(self) -> str:
        """
        Returns a description of the reader.

        :return: the description
        :rtype: str
        """
        return "Counts tokens in text using the specified encoding instance determined from the name of either encoding or model."

    def domains(self) -> List[str]:
        """
        Returns the domains of the filter.

        :return: the domains
        :rtype: list
        """
        return [DOMAIN_PAIRS, DOMAIN_PRETRAIN, DOMAIN_TRANSLATION]

    def accepts(self) -> List:
        """
        Returns the list of classes that are accepted.

        :return: the list of classes
        :rtype: list
        """
        return [PairData, PretrainData, TranslationData]

    def generates(self) -> List:
        """
        Returns the list of classes that get produced.

        :return: the list of classes
        :rtype: list
        """
        return [PairData, PretrainData, TranslationData]

    def _create_argparser(self) -> argparse.ArgumentParser:
        """
        Creates an argument parser. Derived classes need to fill in the options.

        :return: the parser
        :rtype: argparse.ArgumentParser
        """
        parser = super()._create_argparser()
        parser.add_argument("-e", "--encoding", type=str, default=None, help="The name of the encoding to use, e.g., cl100k_base, p50k_base, r50k_base.", required=False)
        parser.add_argument("-m", "--model", type=str, default=None, help="The name of the model to determine the encoding from, e.g., gpt-4, gpt-3.5-turbo, text-davinci-002", required=False)
        parser.add_argument("-p", "--prompt", default=None, type=str, help="The prompt to use with each query (its # of tokens gets added to the total)", required=False)
        parser.add_argument("-t", "--price_per_1k_tokens", metavar="PRICE", default=None, type=float, help="The cost per 1000 tokens", required=False)
        parser.add_argument("-L", "--location", choices=LOCATIONS, default=LOCATION_ANY, help="Which data use for counting tokens; pairs: " + ",".join(LOCATIONS_PAIRS) + ", pretrain: " + ",".join(LOCATIONS_PRETRAIN) + ", translation: " + ",".join(LOCATIONS_PRETRAIN))
        parser.add_argument("-g", "--language", type=str, help="The languages to inspect; inspects all if not specified", required=False, nargs="*")
        return parser

    def _apply_args(self, ns: argparse.Namespace):
        """
        Initializes the object with the arguments of the parsed namespace.

        :param ns: the parsed arguments
        :type ns: argparse.Namespace
        """
        super()._apply_args(ns)
        self.encoding = ns.encoding
        self.model = ns.model
        self.prompt = ns.prompt
        self.price_per_1k_tokens = ns.price_per_1k_tokens
        self.location = ns.location
        self.languages = ns.language

    def initialize(self):
        """
        Initializes the processing, e.g., for opening files or databases.
        """
        super().initialize()

        if self.languages is not None:
            self.languages = [x.lower() for x in self.languages]

        if (self.encoding is None) and (self.model is None):
            raise Exception("Either name of encoding or model must be provided!")
        if self.encoding is not None:
            self._encoding = tiktoken.get_encoding(self.encoding)
        else:
            self._encoding = tiktoken.encoding_for_model(self.model)
        if (self.prompt is not None) and (len(self.prompt) > 0):
            self._count_prompt = self._count_tokens(self.prompt)

    def _count_tokens(self, s: str) -> int:
        """
        Counts the tokens.

        :param s: the string to process
        :type s: str
        :return: the number of tokens
        :rtype: int
        """
        return len(self._encoding.encode(s))

    def process(self, data):
        """
        Processes the data record.

        :param data: the record to process
        :return: the potentially updated record or None if to drop
        """
        result = copy.deepcopy(data)

        if isinstance(result, PairData):
            if self.location in [LOCATION_INSTRUCTION, LOCATION_ANY]:
                self._count += self._count_tokens(result.instruction) + self._count_prompt
            if self.location in [LOCATION_INPUT, LOCATION_ANY]:
                self._count += self._count_tokens(result.input) + self._count_prompt
            if self.location in [LOCATION_OUTPUT, LOCATION_ANY]:
                self._count += self._count_tokens(result.output) + self._count_prompt
        elif isinstance(result, PretrainData):
            if self.location in [LOCATION_CONTENT, LOCATION_ANY]:
                self._count += self._count_tokens(result.content) + self._count_prompt
        elif isinstance(result, TranslationData):
            if self.languages is None:
                for k in result.translations:
                    self._count += self._count_tokens(result.translations[k]) + self._count_prompt
            else:
                for lang in self.languages:
                    if lang in result.translations:
                        self._count += self._count_tokens(result.translations[lang]) + self._count_prompt
        else:
            raise Exception("Unhandled data type: %s" % str(type(result)))

        return result

    def finalize(self):
        """
        Finishes the processing, e.g., for closing files or databases.
        """
        super().finalize()
        if self._count_prompt > 0:
            self.logger().info("# tokens (prompt): %d" % self._count_prompt)
        self.logger().info("# tokens: %d" % self._count)
        if self.price_per_1k_tokens is not None:
            price = self._count / 1000 * self.price_per_1k_tokens
            self.logger().info("total price (1k tokens = %f): %f" % (self.price_per_1k_tokens, price))

# openai-count-tokens

* domain(s): pairs, pretrain, translation
* accepts: ldc.supervised.pairs.PairData, ldc.pretrain.PretrainData, ldc.translation.TranslationData
* generates: ldc.supervised.pairs.PairData, ldc.pretrain.PretrainData, ldc.translation.TranslationData

Counts tokens in text using the specified encoding instance determined from the name of either encoding or model.

```
usage: openai-count-tokens [-h] [-l {DEBUG,INFO,WARN,ERROR,CRITICAL}]
                           [-N LOGGER_NAME] [-e ENCODING] [-m MODEL]
                           [-L {any,instruction,input,output,content}]
                           [-g [LANGUAGE [LANGUAGE ...]]]
                           [-p PRICE_PER_1K_TOKENS]

Counts tokens in text using the specified encoding instance determined from
the name of either encoding or model.

optional arguments:
  -h, --help            show this help message and exit
  -l {DEBUG,INFO,WARN,ERROR,CRITICAL}, --logging_level {DEBUG,INFO,WARN,ERROR,CRITICAL}
                        The logging level to use (default: WARN)
  -N LOGGER_NAME, --logger_name LOGGER_NAME
                        The custom name to use for the logger, uses the plugin
                        name by default (default: None)
  -e ENCODING, --encoding ENCODING
                        The name of the encoding to use, e.g., cl100k_base,
                        p50k_base, r50k_base. (default: None)
  -m MODEL, --model MODEL
                        The name of the model to determine the encoding from,
                        e.g., gpt-4, gpt-3.5-turbo, text-davinci-002 (default:
                        None)
  -L {any,instruction,input,output,content}, --location {any,instruction,input,output,content}
                        Which data use for counting tokens; pairs:
                        any,instruction,input,output, pretrain: any,content,
                        translation: any,content (default: any)
  -g [LANGUAGE [LANGUAGE ...]], --language [LANGUAGE [LANGUAGE ...]]
                        The languages to inspect; inspects all if not
                        specified (default: None)
  -p PRICE_PER_1K_TOKENS, --price_per_1k_tokens PRICE_PER_1K_TOKENS
                        The cost per 1000 tokens (default: None)
```

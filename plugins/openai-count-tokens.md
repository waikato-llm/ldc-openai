# openai-count-tokens

* domain(s): pairs, pretrain, translation, classification
* accepts: ldc.api.supervised.pairs.PairData, ldc.api.pretrain.PretrainData, ldc.api.translation.TranslationData, ldc.api.supervised.classification.ClassificationData
* generates: ldc.api.supervised.pairs.PairData, ldc.api.pretrain.PretrainData, ldc.api.translation.TranslationData, ldc.api.supervised.classification.ClassificationData

Counts tokens in text using the specified encoding instance determined from the name of either encoding or model. When specifying a maximum number of tokens, the filter no longer forwards any data once that threshold has been reached.

```
usage: openai-count-tokens [-h] [-l {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
                           [-N LOGGER_NAME] [--skip] [-e ENCODING] [-m MODEL]
                           [-p PROMPT] [-t PRICE] [-M MAX]
                           [-L [{any,instruction,input,output,content,text} ...]]
                           [-g [LANGUAGE ...]]

Counts tokens in text using the specified encoding instance determined from
the name of either encoding or model. When specifying a maximum number of
tokens, the filter no longer forwards any data once that threshold has been
reached.

options:
  -h, --help            show this help message and exit
  -l {DEBUG,INFO,WARNING,ERROR,CRITICAL}, --logging_level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        The logging level to use. (default: WARN)
  -N LOGGER_NAME, --logger_name LOGGER_NAME
                        The custom name to use for the logger, uses the plugin
                        name by default (default: None)
  --skip                Disables the plugin, removing it from the pipeline.
                        (default: False)
  -e ENCODING, --encoding ENCODING
                        The name of the encoding to use, e.g., cl100k_base,
                        p50k_base, r50k_base. (default: None)
  -m MODEL, --model MODEL
                        The name of the model to determine the encoding from,
                        e.g., gpt-4, gpt-3.5-turbo, text-davinci-002 (default:
                        None)
  -p PROMPT, --prompt PROMPT
                        The prompt to use with each query (its # of tokens
                        gets added to the total) (default: None)
  -t PRICE, --price_per_1k_tokens PRICE
                        The cost per 1000 tokens (default: None)
  -M MAX, --max_tokens MAX
                        The maximum number of tokens to process, unlimited
                        when <1 (default: -1)
  -L [{any,instruction,input,output,content,text} ...], --location [{any,instruction,input,output,content,text} ...]
                        Which data use for counting tokens; classification:
                        any|text, pairs: any|instruction|input|output,
                        pretrain: any|content, translation: any|content
                        (default: any)
  -g [LANGUAGE ...], --language [LANGUAGE ...]
                        The languages to inspect; inspects all if not
                        specified (default: None)
```

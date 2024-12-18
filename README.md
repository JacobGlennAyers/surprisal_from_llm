# surprisal_from_llm
A library for extracting surprisal values from transformers at scale.

This is currently under development. Certain features may be unstable.

Minimal requirements can be found in `requirements.txt` and can be installed via pip: `pip install -r requirements.txt`. Python 3.10 is recommended.

## Usage
Specify an input .csv file with a column of sequences for which you want to generate surprisal values, see 'test.csv' for an example. You also need to specify a name for a huggingface transformer of your choice. Call `python main.py -h` for details on command line args.
```
python main.py --file [input_file] --model [model_name]
```

## Examples

### Usage with GPT2
```
python main.py --file test.csv --model gpt2 --id_column id --keep_space 
```

### Usage with GPT2 calculating entropy
```
python main.py --file test.csv --model gpt2 --id_column id --keep_space --entropy
```


### Usage with BERT and a regex split pattern
```
python main.py --file test.csv --model bert-base-uncased --id_column id --split_pattern "regex[ ']" --batch_size=32 
```

### Usage with an explicitly defined model class
```
python main.py --file test.csv --model flaubert/flaubert_base_cased --id_column id  --model_class FlaubertWithLMHeadModel

```

### Usage with CUDA and GPT2 and a custom output file
```
python main.py --file test.csv --model gpt2 --id_column id --keep_space  --device cuda --outputfile out.csv
```

### Usage with a custom input column
```
python main.py --file [FILE] --model [MODEL] --feature_column [INPUT COLUMN NAME]
```

## Parameters

- `--model REQUIRED`: name of (or path to) a huggingface model you would like to use
- `--file REQUIRED`: path to a .csv input file containing at least one column where each row represents a sequence for which you would like to generate surprisal
- `--split_pattern`: A pattern you would like to use to split your sequence into tokens. By default this gets passed to `.split()`. There are three *special options* available:
  - `regex` Anything following `--split_pattern regex` will be interpreted as a regex pattern and passed to `re.split`. For example, `--split_pattern regex[ \?\!\.]` will result in a regex that splits on whitespace and ".?!". It is recommended to represent space as " " rather than "\s".
  - `tokenizer` Using `--split_pattern tokenizer` will allow the tokenizer to decide what constitutes a word. This only works with FastTokenizers. A fast tokenizer will be automatically selected if available.
  - `token` Using `--split_pattern token` will generate surprisal values for the actual tokens the tokenizer generates in its tokenization step. With subword tokenizers this can result in multiple tokens per orthographic word.
- `--outputfile`: The path and name to your outputfile, defaults to `features.csv`
- `--feature_column`: The name of the column containing your sequences in your input file. Defaults to 'sentences'
- `--id_column`: If your input file contains a column with ids for your sequences, then you can specify this column here so your ids carry over to your input file. If left empty, the module with generate ids for you.
- `--device`: Allows you to specify the device to run the model on. Defaults to 'cpu'
- `--keep_space`: Optional flag that is necessary when running a custom split pattern on tokenizers that differentiate between tokens preceded by whitespace and those that do not.
- `--batch_size`: Batch size for inference
- `--model_class`: Allows you to specify a model class. By default this library uses the AutoModelForCausalLM class. Some models may not work with this class. In this case you will get a 'Unrecognized configuration class' error. Model class has to be a type of LM Model for this library to work.
  - example usage: `--model_class FlaubertWithLMHeadModel`
- `--tokenizer_class`: Allows you to specify a tokenizer class. By default this library uses the AutoTokenizer class. Some models may not work with this class. In this case you will get a 'Unrecognized configuration class' error. 
- `--no_bos`: Tells model not to include a bos-token. Note that this means there will be no surprisal values for the first token in the sequence. This setting is useful for models that have no bos_token. Note that if this is selected, the value for the first word is returned as 0.0.
- `--sep`: Allows to specify a separation character for your input file.
- `--entropy`: Tells module to calculate conditioned entropy in addition to surprisal. Whereby entropy is defined as Shannon-Entropy:

$`{H(X) :=-\sum\limits_{x\in{\mathcal{X}}} p(x)\log p(x)}`$


- `--logbase`: Specifies the logbase, options are 2, e, or 10. Default = 2


## Usage for Token Classification based Surprisal
Token classification surprisal can be extracted by calling the `main_classification.py` file. This shares all parameters with `main.py` except for an additional `--tag_column` param that needs to be pointed at the column in your file containing your classification tags.

This is agnostic to the type of classification task performed. The important thing to consider is that the ground truth tags in your file have to be tags that are in the models list of tags. If you do not know what tags your model uses, you can call `model.config.label2id` in a notebook or ipython environment to get a list of permissible tags.

Please note that currently your tags need to be separated by whitespace and you need to ensure that you have exactly as many tags as you have words based on your selected tokenization strategy (default: whitespace).

```
python main_classification.py --file [input_file] --model [model_name]
```

### Example Usage for Token Classification

```
python main_classifier.py --file classification_test.txt --model "QCRI/bert-base-multilingual-cased-pos-english"
```

## Additional Params for Classification

- `--tag_column`: name of the column where your classification tags can be found. Default = 'tags'
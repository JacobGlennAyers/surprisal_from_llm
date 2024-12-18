import surprisal_classification as sp
import transformers
import pandas as pd
import argparse
import utils as ut
from transformers import AutoTokenizer, AutoModelForTokenClassification


def start_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Reads in a column in a .csv file as a sequence of strings. Produces '
                                                 'a .csv file with transformer based surprisal values for the sequences'
                                                 '. Each line in the original file is read in as a sequence')
    parser.add_argument("--model",
                        help="The model name within the transformer library. For example, gpt2",
                        required=False)
    parser.add_argument("--file",
                        help="The path to the file you wish to process.",
                        required=False)
    parser.add_argument("--split_pattern",
                        help="The pattern you want to use to split your tokens on. This defaults to whitespace. If you "
                             "want to define a regex pattern, begin your pattern with a 'regex' flag like so:"
                             r" regex[ '\!]"
                             "\nSpecial Options are: \n"
                             "'tokenizer': let the tokenizer define what constitutes a word \n "
                             "'token': use the tokenizer tokenization, this might result in subword tokens",
                        required=False, default=None)
    parser.add_argument("--outputfile",
                        help="The name of your outputfile. Default: features.csv",
                        required=False, default='features.csv')
    parser.add_argument("--feature_column",
                        help="The name of your feature_column in the file. "
                             "This column contains the sentences over which"
                             " you want to produce surprisal values. Default: sentences",
                        required=False, default='sentences')
    parser.add_argument("--tag_column",
                        help="The name of your tag column in the file. "
                             "This column contains the tags over which"
                             " you want to produce surprisal values."
                             " The tags should be separated by whitespace Default: tags",
                        required=False, default='tags')
    parser.add_argument("--id_column",
                        help="The name of your id_column, if not specified the module will generate its own ids."
                             "If an id_column is specified the module will use this column to generate sequence ids",
                        required=False, default=None)
    parser.add_argument("--sep",
                        help="The separation character for your .csv file",
                        required=False, default=',')
    parser.add_argument("--device",
                        help="Device on which to run your model. Default: cpu",
                        required=False, default='cpu')
    parser.add_argument("--keep_space",
                        help="This flag is necessary for tokenizers that distinguish between tokens with a preceding"
                             " space and those without",
                        required=False, action='store_true')
    parser.add_argument("--batch_size",
                        help="Regulates the batch size passed to the model, try decreasing if you run into a memory "
                             "error. Default=64",
                        required=False, default=64, type=int)
    parser.add_argument("--model_class",
                        help="Allows you to specify a model class. "
                             "By default this library uses the AutoModelForCausalLM class. Some models may not work"
                             " with this class. In this case you will get a 'Unrecognized configuration class' error."
                             " Model class has to be a type of LM Model for this library to work"
                             " example usage: --model_class FlaubertWithLMHeadModel",
                        required=False, default=None)
    parser.add_argument("--tokenizer_class",
                        help="Allows you to specify a model class. "
                             "By default this library uses the AutoTokenizer class. Some models may not work"
                             " with this class. In this case you will get a 'Unrecognized configuration class' error."
                             " Model class has to be a type of LM Model for this library to work"
                             " example usage: --token_class GPT2Tokenizer",
                        required=False, default=None)
    parser.add_argument("--no_bos",
                        help="Tells model not to include a bos-token. Note that this means there will be no surprisal "
                             " values for the first token in the sequence",
                        required=False, action='store_true')
    parser.add_argument("--hf_token",
                        help="your huggingface access token",
                        required=False, default=None)
    parser.add_argument("--entropy",
                        help="Tells model to also calculate and return Shannon entropy",
                        required=False, action='store_true')
    parser.add_argument('--logbase',
                        choices=['2', 'e', '10'],
                        help='Specifies the logbase, options are 2, e, or 10. Default = 2',
                        required=False, default='2')

    return parser


def main(sequence, tag_sequence, model, tokenizer, id_sequence=None, batch_size=64, no_bos=False, entropy=False, logbase='2',
         **kwargs):
    cols = ['id', 'batch_idx', 'sequ_idx', 'word_idx', 'word', 'tag', 'surprisal', 'entropy'] if entropy \
        else ['id', 'batch_idx', 'sequ_idx', 'word_idx', 'word', 'tag', 'surprisal']
    df = pd.DataFrame(columns=cols)
    print("starting")
    processor = sp.batch_process(sequence, tag_sequence, model, tokenizer, batch_size=batch_size, no_bos=no_bos,
                                 entropy=entropy,
                                 logbase=logbase,
                                 **kwargs)
    # used to increment the index for the original id column if one is passed
    id_idx = 0
    for batch_idx, output in enumerate(processor):
        _, batch_list = output
        for sequ_idx, sequence in enumerate(batch_list):
            for i in range(len(sequence)):
                if entropy:
                    word, tag, surprisal, entropies, _ = sequence[i]
                else:
                    word, tag, surprisal, _ = sequence[i]
                if id_sequence:
                    id_unique = str(batch_idx) + '_' + str(id_sequence[id_idx]) + '_' + str(i)
                    sequ_id = id_sequence[id_idx]
                else:
                    id_unique = str(batch_idx) + '_' + str(sequ_idx) + '_' + str(i)
                    sequ_id = sequ_idx
                data_entry = (id_unique, batch_idx, sequ_id, i, word, tag, surprisal, entropies) if entropy \
                    else (id_unique, batch_idx, sequ_id, i, word, tag, surprisal)
                row = {i: j for i, j in zip(cols, data_entry, strict=True)}
                df.loc[len(df)] = row
            id_idx += 1
    return df


if __name__ == "__main__":
    parser = start_parser()
    arguments = parser.parse_args()

    data_in, tags_in, ids = ut.read_in_csv_classifier(arguments.file, arguments.feature_column, arguments.tag_column,
    arguments.id_column, sep=arguments.sep)
    if arguments.model_class:
        model_class = getattr(transformers, arguments.model_class)
        model = model_class.from_pretrained(arguments.model, token=arguments.hf_token).to(arguments.device)
    else:
        model = AutoModelForTokenClassification.from_pretrained(arguments.model, token=arguments.hf_token).to(arguments.device)

    if arguments.tokenizer_class:
        tokenizer_class = getattr(transformers, arguments.tokenizer_class)
        tokenizer = tokenizer_class.from_pretrained(arguments.model, token=arguments.hf_token)
    else:
        tokenizer = AutoTokenizer.from_pretrained(arguments.model, token=arguments.hf_token)

    if arguments.keep_space:
        keep_space = True
    else:
        keep_space = False

    data = main(data_in, tags_in, model, tokenizer, id_sequence=ids, padding=True, keep_space=keep_space,
                device=arguments.device,
                sum=True,
                split_pattern=arguments.split_pattern,
                batch_size=arguments.batch_size,
                no_bos=arguments.no_bos,
                entropy=arguments.entropy,
                logbase=arguments.logbase)

    data.to_csv(arguments.outputfile)

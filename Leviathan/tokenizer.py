""" These classes are pretty useful. Later, we'll add a bpe model that we can train using YouTokenToMe. """

from Leviathan.imports import *

class Tokenizer:
    """ Vanilla Tokenizer. Will train a BPE model and use that. """
    def __init__(self, text) -> None:
        self._text = text
        self._chars = sorted(list(set(self._text)))
        self._vocab_size = len(self._chars)

    def _string_to_integer(self):
        """ Utility function for creating a mapping from character to integers. Converts string to integer. """
        return {ch: i for i, ch in enumerate(self._chars)}


    def _integer_to_string(self):
        """ Utility function for creating a mapping from integers to character. Converts integer to string. """
        return {ch: i for i, ch in enumerate(self._chars)}


    def encode(self):
        """ Encode the text, and returns the list of integers. """
        warnings.warn('This is a pretty naive methos to create a mapping, and will not be scalable once large enough dataset is there. ', DeprecationWarning)
        return [self._string_to_integer(char) for char in self._text]


    def decode(self, lst_ints:list[int]):
        """ Decode the list of integers to convert them into string. Useful to see how the model understands things. """
        return ''.join([self._integer_to_string(i) for i in lst_ints])


class BPETokenizer:
    """ Improving upon the basic vanilla Tokenizer. """
    def __init__(self, path_to_file) -> None:
        pass
    
    def train_bpe(self, model_name:str):
        """ Trains a BPE model on the file. Reads the path, gets the text, and generate the tokens. """
        pass

    def encode(self, flags):
        """ Encodes the text. """
        pass

    def decode(self, flags):
        """ Decodes the text. """
        pass


class ReadDataAndModify:
    """ Read the data from, and  modify it however you like. """
    def __init__(self, file_path, is_url=False) -> None:
        self._file_path = file_path
        self._is_url = is_url

    def _get_data_from_file(self):
        if self._is_url:
            raise NotImplementedError
        else:
            with open(self._file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        return text

    def _return_tokenized_input(self):
        text = self._get_data_from_file()
        tokenizer = Tokenizer(text)
        tokenized_input = tokenizer.encode()

        return tokenized_input
            
    def _convert_list_to_tensor(self, tokenized_input, data_type=torch.long):
         return torch.tensor(tokenized_input, dtype=data_type)

    def _generate_splits(data:torch.Tensor, train_split:float):
        n = int(train_split*len(data))

        return data[:n], data[n:]
            
    def split_data_test_train(self, train_split=0.9):
        tokenized_input = self._return_tokenized_input()

        data = self._convert_list_to_tensor(tokenized_input)

        return self._generate_splits(data, train_split)

import logging
logging.basicConfig(
        format="%(asctime)s — %(levelname)s — %(name)s — %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

from aitextgen import aitextgen
from aitextgen.colab import mount_gdrive, copy_file_from_gdrive
from aitextgen.TokenDataset import TokenDataset, merge_datasets
from aitextgen.utils import build_gpt2_config
from aitextgen.tokenizers import train_tokenizer

ai = aitextgen(model='flyhero/gpt-j-6B', to_gpu=True)

test = ai.generate(prompt='J K Rowling Author Biography', n=1, max_length=32)

print(test)

import time
import torch
from transformers import BertTokenizer
import modeling


def main(cfg, text):
    start = time.time()

    device = torch.device('cpu')
    print(f'device = {device}')

    tokenizer = BertTokenizer.from_pretrained('emeraldgoose/bad-korean-tokenizer')
    vocab_size = tokenizer.vocab_size + len(tokenizer.get_added_vocab())
    
    model = modeling.Model(vocab_size=vocab_size, max_seq=cfg['max_seq'], embedding_dim=cfg['embedding_dim'], channel=cfg['channel'], num_class=2)
    model.to(device)
    
    input = tokenizer(
        text, 
        return_tensors='pt',
        padding='max_length',
        truncation=True,
        max_length=200)['input_ids'].to(device)
    
    model.eval()
    output = model(input)
    output = output.argmax(-1)

    end = time.time()

    print(('Hate' if output == 1 else 'None'), f'Inference time : {end-start}')


if __name__ == "__main__":
    cfg = dict(max_seq=200, embedding_dim=128, channel=256)
    text = '안녕하세요'
    main(cfg, text)
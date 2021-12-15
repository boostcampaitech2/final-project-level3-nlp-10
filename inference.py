import time
import torch
from transformers import BertTokenizer
import modeling


def main(config, text):
    device = torch.device('cpu')
    print(f'device = {device}')

    tokenizer = BertTokenizer.from_pretrained('emeraldgoose/bad-korean-tokenizer')
    vocab_size = tokenizer.vocab_size + len(tokenizer.get_added_vocab())
    
    model = modeling.Model(
        vocab_size=vocab_size, 
        embedding_dim=config['embedding_dim'], 
        channel=config['channel'], 
        num_class=2,
        dropout1=config['dropout1'],
        dropout2=config['dropout2'],
        device=device)
    model.load_state_dict(torch.load('save/temp/result.pt'))
    model.to(device)
    
    start = time.time()

    input = tokenizer(
        text, 
        return_tensors='pt',
        padding='max_length',
        truncation=True,
        max_length=200,
        return_token_type_ids=False)['input_ids'].to(device)
    
    model.eval()
    output = model(input)
    output = output.argmax(-1)

    end = time.time()

    print('Result : ', 'Hate' if output == 1 else 'None',)
    print(f'Inference time : {end-start}')


if __name__ == "__main__":
    cfg = dict(embedding_dim=100, channel=32, num_class=2, dropout1=0.3, dropout2=0.5)
    text = '유하zzzzzz'
    main(cfg, text)
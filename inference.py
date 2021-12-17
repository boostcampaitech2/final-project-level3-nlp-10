import time
import torch
from transformers import BertTokenizer
import modeling


def main(config, text):
    device = torch.device('cpu')
    print(f'device = {device}')

    tokenizer = BertTokenizer.from_pretrained('jiho0304/bad-korean-tokenizer')
    vocab_size = tokenizer.vocab_size + len(tokenizer.get_added_vocab())
    
    model = modeling.Model(
        vocab_size=vocab_size, 
        embedding_dim=config['embedding_dim'], 
        channel=config['channel'], 
        num_class=2,
        dropout1=config['dropout1'],
        dropout2=config['dropout2'],
        device='cpu')
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
    m = torch.nn.Softmax(dim=1)
    prob = m(output)
    output = output.argmax(-1)

    end = time.time()

    print('Result : ', 'Hate' if output == 1 else 'None')
    print(f'Probability : {prob[0][output].item():.2f}')
    print(f'Inference time : {end-start}')


if __name__ == "__main__":
    cfg = dict(embedding_dim=100, channel=128, num_class=2, dropout1=0.3, dropout2=0.5)
    text = ''
    main(cfg, text)
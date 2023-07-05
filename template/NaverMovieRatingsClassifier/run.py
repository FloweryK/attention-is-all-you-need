import os
import sys
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from config import Config
from vocab import VocabSPM
from dataset import RatingsDataset, collate_fn
from classifier import RatingsClassifier
from trainer import Trainer


if __name__ == '__main__':
    # vocab
    vocab = VocabSPM('src/vocab/spm/kowiki_8000.model')

    # config
    config = Config(
        n_vocab=len(vocab),
        n_seq=200,
        n_layer=6,
        n_head=8,
        d_emb=128,
        d_hidden=100,
        scale=100**(1/2),
        n_epoch=100,
        n_batch=64,
        n_output=2,
        device='cpu',
        lr=0.0001,
    )

    # dataset
    trainset = RatingsDataset(vocab, 'template/NaverMovieRatingsClassifier/data/ratings_train.txt')
    testset = RatingsDataset(vocab, 'template/NaverMovieRatingsClassifier/data/ratings_test.txt')
    trainloader = DataLoader(trainset, batch_size=config.n_batch, shuffle=True, collate_fn=collate_fn)
    testloader = DataLoader(testset, batch_size=config.n_batch, shuffle=False, collate_fn=collate_fn)

    # model
    model = RatingsClassifier(config)

    # trainer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    trainer = Trainer(config, model, criterion, optimizer)

    # train
    for epoch in range(config.n_epoch):
        trainer.run_epoch(epoch, trainloader, device=config.device, train=True)
        trainer.run_epoch(epoch, testloader, device=config.device, train=False)
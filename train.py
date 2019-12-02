import os
from collections import OrderedDict

import torch
import torch.optim
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from vocab import Vocabulary, load_glove
from build_model import build_model, initialize_model_
from reader_batcher import sst_reader, prepare_minibatch, get_minibatch
from evaluater import evaluate


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print("device:", device)


def train():
    
    ### CONFIGS ###

    batch_size = 25
    lr = 0.0002
    weight_decay = 1e-5
    
    iter_i = 0
    train_loss = 0.
    losses = []
    accuracies = []
    best_eval = 1.0e9
    best_iter = 0
    threshold = 1e-4
    
    ################
    
    train_data = list(sst_reader("../../data/sst/train.txt"))
    dev_data = list(sst_reader("../../data/sst/dev.txt"))
    test_data = list(sst_reader("../../data/sst/test.txt"))

    iters_per_epoch = len(train_data) // batch_size

    vocabulary = Vocabulary()
    glove = load_glove('../../data/sst/glove.840B.300d.sst.txt', vocabulary)
    t2i = OrderedDict({p: i for p, i in zip(["very negative", "negative", "neutral", "positive", "very positive"], range(5))})

    # Build model
    model = build_model(vocabulary, t2i)
    initialize_model_(model)  # интересно, но без инициализации модель не сойдется

    with torch.no_grad():
        model.embed.weight.data.copy_(torch.from_numpy(glove))
        model.embed.weight.requires_grad = False
        model.embed.weight[1] = 0.0

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5,
                                  verbose=True, cooldown=5, threshold=1e-4, min_lr=1e-5)

    model = model.to(device)

    while True:  # when we run out of examples, shuffle and continue
        for batch in get_minibatch(train_data, batch_size=batch_size, shuffle=True):
            epoch = iter_i // iters_per_epoch

            model.train()
            x, targets, _ = prepare_minibatch(batch, model.vocab, device=device)
            mask = (x != 1)

            logits = model(x)

            loss, loss_optional = model.get_loss(logits, targets, mask=mask)
            model.zero_grad()

            train_loss += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.)
            optimizer.step()
            iter_i += 1
            
            if iter_i % 100 == 0:

                train_loss = train_loss / 100
                print(f"Epoch {epoch}, Iter {iter_i}, loss={train_loss}")
                losses.append(train_loss)
                train_loss = 0.


            if iter_i % iters_per_epoch == 0:
                dev_eval = evaluate(model, dev_data, batch_size=batch_size, device=device)

                print(f"Epoch {epoch} iter {iter_i}: dev {dev_eval['acc']}")

                # save best model parameters
                compare_score = dev_eval["loss"]
                if "obj" in dev_eval:
                    compare_score = dev_eval["obj"]

                scheduler.step(compare_score)  # adjust learning rate

                if (compare_score < (best_eval * (1-threshold))) and iter_i > (3 * iters_per_epoch):

                    best_eval = compare_score
                    best_iter = iter_i

                    if not os.path.exists('sst_results/default'):
                        os.makedirs('sst_results/default')

                    ckpt = {
                        "state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_eval": best_eval,
                        "best_iter": best_iter
                    }
                    path = os.path.join('sst_results/default', "model.pt")
                    torch.save(ckpt, path)


            if iter_i == (iters_per_epoch * 25):

                path = os.path.join('sst_results/model', "model.pt")
                model.load_state_dict(torch.load(path)["state_dict"])

                print("# Evaluating")
                dev_eval = evaluate(model, dev_data, batch_size=25, device=device)
                test_eval = evaluate(model, test_data, batch_size=25, device=device)

                print("best model iter {:d}: "
                      "dev {} test {}".format(best_iter, dev_eval['acc'], test_eval['acc']))

                return losses, accuracies
            
train()
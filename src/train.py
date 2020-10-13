from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"
optimizer = optim.Adam(model.parameters(), weight_decay=0)
criterion = nn.CrossEntropyLoss().to(device)

hyperparams = {'vocab_size' : len(vocab_dict),
               'embedding_dim' : 75,
               'n_filters' : 3,
               'filter_sizes' : [2, 3, 4],
               'ngram':3,
               'output_dim' : len(label_dict),
               'dropout' : 0.5,
               'pad_idx' : vocab_dict["^PAD"],
               'batch_size' : 10,
              'device':device}

def calc_acc(logit, real):
    pred = logit.argmax(1)
    correct = (pred == real).sum()
    return float(correct.cpu().data) / logit.shape[0]


def calc_loss_c(model, y, device):
    """
    torch.tensor([0,1,2]) is decoded identity label vector
    """
    f2_c = model.fc(model.compat_model.c)
    return criterion(f2_c, torch.range(0, y.shape[1] - 1, dtype=torch.long).to(device))


def fit(batch_size=hyperparams['batch_size'],
        device=hyperparams['device'],
        epoch=5):
    for proc in xrange(epoch):
        acc_ls = []
        loss_ls = []

        shuffle_idx = torch.randint(high=batch_size, size=(train_X.shape[0],), device=hyperparams['device'])
        use_X = train_X[shuffle_idx]
        use_y = train_y[shuffle_idx]

        for batch_idx in tqdm(xrange(0, use_X.shape[0], batch_size)):
            model.train()
            batch_X = use_X[batch_idx: batch_idx + batch_size]
            batch_y = use_y[batch_idx: batch_idx + batch_size]

            if batch_X.shape[0] != batch_size: continue
            logit = model(batch_X)

            label = batch_y.argmax(1).clone()

            optimizer.zero_grad()

            loss_v = criterion(logit, label)
            loss_c = calc_loss_c(model, batch_y, device)
            loss = loss_v + loss_c

            acc = calc_acc(logit, label)
            loss.backward()
            optimizer.step()

            acc_ls.append(acc)
            loss_ls.append(float(loss.cpu().data))

        print("EPOCH : {} | ACC : {} | LOSS : {}".format(proc + 1, np.mean(acc_ls), np.mean(loss_ls)))

        # validate
        acc_ls = []
        loss_ls = []

        model.eval()
        for batch_idx in tqdm(xrange(0, test_X.shape[0], batch_size)):

            batch_X = test_X[batch_idx: batch_idx + batch_size]
            batch_y = test_y[batch_idx: batch_idx + batch_size]
            if batch_X.shape[0] != batch_size: continue
            logit = model(batch_X)
            label = batch_y.argmax(1).clone()

            optimizer.zero_grad()
            loss = criterion(logit, label)

            acc = calc_acc(logit, label)
            loss.backward()
            optimizer.step()

            acc_ls.append(acc)
            loss_ls.append(float(loss.cpu().data))

        print("EPOCH : {} | VAL ACC : {} | VAL LOSS : {}".format(proc + 1, np.mean(acc_ls), np.mean(loss_ls)))

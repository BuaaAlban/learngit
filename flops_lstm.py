
def count_lstm(m, x, y):


    hidden_size = m.hidden_size

    num_layers = m.num_layers

    seq_len, batch_size, nfeatures = x.size()

    total_ops =8*hidden_size*(hidden_size + nfeatures)*seq_len*batch_size*num_layers

    m.total_ops = torch.Tensor([int(total_ops)])

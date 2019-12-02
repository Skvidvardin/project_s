import torch
import numpy as np
import math
from lstm_encoder import LSTMEncoder
from kuma_gate import KumaGate

from torch.nn.init import _calculate_fan_in_and_fan_out
from torch import nn

def get_z_stats(z=None, mask=None):
    """
    Computes statistics about how many zs are
    exactly 0, continuous (between 0 and 1), or exactly 1.

    :param z:
    :param mask: mask in [B, T]
    :return:
    """

    z = torch.where(mask, z, z.new_full([1], 1e2))

    num_0 = (z == 0.).sum().item()
    num_c = ((z > 0.) & (z < 1.)).sum().item()
    num_1 = (z == 1.).sum().item()

    total = num_0 + num_c + num_1
    mask_total = mask.sum().item()

    assert total == mask_total, "total mismatch"
    return num_0, num_c, num_1, mask_total


class Classifier(nn.Module):
    """
    The Encoder takes an input text (and rationale z) and computes p(y|x,z)

    Supports a sigmoid on the final result (for regression)
    If not sigmoid, will assume cross-entropy loss (for classification)

    """

    def __init__(self,
                 embed:        nn.Embedding = None,
                 hidden_size:  int = 200,
                 output_size:  int = 1,
                 dropout:      float = 0.1,
                 layer:        str = "rcnn",
                 nonlinearity: str = "sigmoid"
                 ):

        super(Classifier, self).__init__()

        emb_size = embed.weight.shape[1]

        self.embed_layer = nn.Sequential(
            embed,
            nn.Dropout(p=dropout)
        )

        self.enc_layer = LSTMEncoder(emb_size, hidden_size, bidirectional=True)
        enc_size = hidden_size * 2

        self.output_layer = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(enc_size, output_size),
            nn.Sigmoid() if nonlinearity == "sigmoid" else nn.LogSoftmax(dim=-1)
        )

        self.report_params()

    def report_params(self):
        count = 0
        for name, p in self.named_parameters():
            if p.requires_grad and "embed" not in name:
                count += np.prod(list(p.shape))
        print("{} #params: {}".format(self.__class__.__name__, count))

    def forward(self, x, mask, z=None):

        rnn_mask = mask
        emb = self.embed_layer(x)

        # apply z to main inputs
        if z is not None:
            z_mask = (mask.float() * z).unsqueeze(-1)  # [B, T, 1]
            rnn_mask = z_mask.squeeze(-1) > 0.  # z could be continuous
            emb = emb * z_mask

        # z is also used to control when the encoder layer is active
        lengths = mask.long().sum(1)

        # encode the sentence
        _, final = self.enc_layer(emb, rnn_mask, lengths)

        # predict sentiment from final state(s)
        y = self.output_layer(final)

        return y


class IndependentLatentModel(nn.Module):
    """
    The latent model ("The Generator") takes an input text
    and returns samples from p(z|x)
    This version uses a reparameterizable distribution, e.g. HardKuma.
    """

    def __init__(self,
                 embed:        nn.Embedding = None,
                 hidden_size:  int = 200,
                 dropout:      float = 0.1,
                 layer:        str = "rcnn",
                 distribution: str = "kuma"
                 ):

        super(IndependentLatentModel, self).__init__()

        self.layer = layer
        emb_size = embed.weight.shape[1]
        enc_size = hidden_size * 2

        self.embed_layer = nn.Sequential(embed, nn.Dropout(p=dropout))
        self.enc_layer = LSTMEncoder(emb_size, hidden_size, bidirectional=True)

        if distribution == "kuma":
            self.z_layer = KumaGate(enc_size)
        else:
            raise ValueError("unknown distribution")

        self.z = None
        self.z_dists = []

        self.report_params()

    def report_params(self):
        count = 0
        for name, p in self.named_parameters():
            if p.requires_grad and "embed" not in name:
                count += np.prod(list(p.shape))
        print("{} #params: {}".format(self.__class__.__name__, count))

    def forward(self, x, mask, **kwargs):

        # encode sentence
        lengths = mask.sum(1)

        emb = self.embed_layer(x)  # [B, T, E]
        h, _ = self.enc_layer(emb, mask, lengths)

        z_dist = self.z_layer(h)

        # we sample once since the state was already repeated num_samples
        if self.training:
            if hasattr(z_dist, "rsample"):
                z = z_dist.rsample()  # use rsample() if it's there
            else:
                z = z_dist.sample()  # [B, M, 1]
        else:
            # deterministic strategy
            p0 = z_dist.pdf(h.new_zeros(()))
            p1 = z_dist.pdf(h.new_ones(()))
            pc = 1. - p0 - p1  # prob. of sampling a continuous value [B, M]
            z = torch.where(p0 > p1, h.new_zeros([1]),  h.new_ones([1]))
            z = torch.where((pc > p0) & (pc > p1), z_dist.mean(), z)  # [B, M, 1]

        # mask invalid positions
        z = z.squeeze(-1)
        z = torch.where(mask, z, z.new_zeros([1]))

        self.z = z  # [B, T]
        self.z_dists = [z_dist]

        return z


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


__all__ = ["LatentRationaleModel"]


class LatentRationaleModel(nn.Module):
    """
    Latent Rationale
    Categorical output version (for SST)

    Consists of:

    p(y | x, z)     observation model / classifier
    p(z | x)        latent model

    """
    def __init__(self,
                 vocab:          object = None,
                 vocab_size:     int = 0,
                 emb_size:       int = 200,
                 hidden_size:    int = 200,
                 output_size:    int = 1,
                 dropout:        float = 0.1,
                 layer:          str = "lstm",
                 dependent_z:    bool = False,
                 z_rnn_size:     int = 30,
                 selection:      float = 1.0,
                 lasso:          float = 0.0,
                 lambda_init:    float = 1e-4,
                 lagrange_lr:    float = 0.01,
                 lagrange_alpha: float = 0.99,
                 ):

        super(LatentRationaleModel, self).__init__()

        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.vocab = vocab
        self.selection = selection
        self.lasso = lasso

        self.z_rnn_size = z_rnn_size
        self.dependent_z = dependent_z

        self.embed = embed = nn.Embedding(vocab_size, emb_size, padding_idx=1)

        self.classifier = Classifier(
            embed=embed, hidden_size=hidden_size, output_size=output_size,
            dropout=dropout, layer=layer, nonlinearity="softmax")

        self.latent_model = IndependentLatentModel(
            embed=embed, hidden_size=hidden_size,
            dropout=dropout, layer=layer)

        self.criterion = nn.NLLLoss(reduction='none')

        # lagrange
        self.lagrange_alpha = lagrange_alpha
        self.lagrange_lr = lagrange_lr
        self.register_buffer('lambda0', torch.full((1,), lambda_init))
        self.register_buffer('lambda1', torch.full((1,), lambda_init))
        self.register_buffer('c0_ma', torch.full((1,), 0.))
        self.register_buffer('c1_ma', torch.full((1,), 0.))

    @property
    def z(self):
        return self.latent_model.z

    @property
    def z_layer(self):
        return self.latent_model.z_layer

    @property
    def z_dists(self):
        return self.latent_model.z_dists

    def predict(self, logits, **kwargs):
        """
        Predict deterministically.
        :param x:
        :return: predictions, optional (dict with optional statistics)
        """
        assert not self.training, "should be in eval mode for prediction"
        return logits.argmax(-1)

    def forward(self, x):
        """
        Generate a sequence of zs with the Generator.
        Then predict with sentence x (zeroed out with z) using Encoder.

        :param x: [B, T] (that is, batch-major is assumed)
        :return:
        """
        mask = (x != 1)  # [B,T]
        z = self.latent_model(x, mask)
        y = self.classifier(x, mask, z)

        return y

    def get_loss(self, logits, targets, mask=None, **kwargs):

        optional = {}
        selection = self.selection
        lasso = self.lasso

        loss_vec = self.criterion(logits, targets)  # [B]

        # main MSE loss for p(y | x,z)
        ce = loss_vec.mean()        # [1]
        optional["ce"] = ce.item()  # [1]

        batch_size = mask.size(0)
        lengths = mask.sum(1).float()  # [B]

        # z = self.generator.z.squeeze()
        z_dists = self.latent_model.z_dists

        # pre-compute for regularizers: pdf(0.)
        if len(z_dists) == 1:
            pdf0 = z_dists[0].pdf(0.)
        else:
            pdf0 = []
            for t in range(len(z_dists)):
                pdf_t = z_dists[t].pdf(0.)
                pdf0.append(pdf_t)
            pdf0 = torch.stack(pdf0, dim=1)  # [B, T, 1]

        pdf0 = pdf0.squeeze(-1)
        pdf0 = torch.where(mask, pdf0, pdf0.new_zeros([1]))  # [B, T]

        # L0 regularizer
        pdf_nonzero = 1. - pdf0  # [B, T]
        pdf_nonzero = torch.where(mask, pdf_nonzero, pdf_nonzero.new_zeros([1]))

        l0 = pdf_nonzero.sum(1) / (lengths + 1e-9)  # [B]
        l0 = l0.sum() / batch_size

        # `l0` now has the expected selection rate for this mini-batch
        # we now follow the steps Algorithm 1 (page 7) of this paper:
        # https://arxiv.org/abs/1810.00597
        # to enforce the constraint that we want l0 to be not higher
        # than `selection` (the target sparsity rate)

        # lagrange dissatisfaction, batch average of the constraint
        c0_hat = (l0 - selection)

        # moving average of the constraint
        self.c0_ma = self.lagrange_alpha * self.c0_ma + \
            (1 - self.lagrange_alpha) * c0_hat.item()

        # compute smoothed constraint (equals moving average c0_ma)
        c0 = c0_hat + (self.c0_ma.detach() - c0_hat.detach())

        # update lambda
        self.lambda0 = self.lambda0 * torch.exp(self.lagrange_lr * c0.detach())

        with torch.no_grad():
            optional["cost0_l0"] = l0.item()
            optional["target0"] = selection
            optional["c0_hat"] = c0_hat.item()
            optional["c0"] = c0.item()  # same as moving average
            optional["lambda0"] = self.lambda0.item()
            optional["lagrangian0"] = (self.lambda0 * c0_hat).item()
            optional["a"] = z_dists[0].a.mean().item()
            optional["b"] = z_dists[0].b.mean().item()

        loss = ce + self.lambda0.detach() * c0

        if lasso > 0.:
            # fused lasso (coherence constraint)

            # cost z_t = 0, z_{t+1} = non-zero
            zt_zero = pdf0[:, :-1]
            ztp1_nonzero = pdf_nonzero[:, 1:]

            # cost z_t = non-zero, z_{t+1} = zero
            zt_nonzero = pdf_nonzero[:, :-1]
            ztp1_zero = pdf0[:, 1:]

            # number of transitions per sentence normalized by length
            lasso_cost = zt_zero * ztp1_nonzero + zt_nonzero * ztp1_zero
            lasso_cost = lasso_cost * mask.float()[:, :-1]
            lasso_cost = lasso_cost.sum(1) / (lengths + 1e-9)  # [B]
            lasso_cost = lasso_cost.sum() / batch_size

            # lagrange coherence dissatisfaction (batch average)
            target1 = lasso

            # lagrange dissatisfaction, batch average of the constraint
            c1_hat = (lasso_cost - target1)

            # update moving average
            self.c1_ma = self.lagrange_alpha * self.c1_ma + \
                (1 - self.lagrange_alpha) * c1_hat.detach()

            # compute smoothed constraint
            c1 = c1_hat + (self.c1_ma.detach() - c1_hat.detach())

            # update lambda
            self.lambda1 = self.lambda1 * torch.exp(
                self.lagrange_lr * c1.detach())

            with torch.no_grad():
                optional["cost1_lasso"] = lasso_cost.item()
                optional["target1"] = lasso
                optional["c1_hat"] = c1_hat.item()
                optional["c1"] = c1.item()  # same as moving average
                optional["lambda1"] = self.lambda1.item()
                optional["lagrangian1"] = (self.lambda1 * c1_hat).item()

            loss = loss + self.lambda1.detach() * c1

        # z statistics
        if self.training:
            num_0, num_c, num_1, total = get_z_stats(self.latent_model.z, mask)
            optional["p0"] = num_0 / float(total)
            optional["pc"] = num_c / float(total)
            optional["p1"] = num_1 / float(total)
            optional["selected"] = 1 - optional["p0"]

        return loss, optional


def xavier_uniform_n_(w, gain=1., n=4):
    """
    Xavier initializer for parameters that combine multiple matrices in one
    parameter for efficiency. This is e.g. used for GRU and LSTM parameters,
    where e.g. all gates are computed at the same time by 1 big matrix.
    :param w:
    :param gain:
    :param n:
    :return:
    """
    with torch.no_grad():
        fan_in, fan_out = _calculate_fan_in_and_fan_out(w)
        assert fan_out % n == 0, "fan_out should be divisible by n"
        fan_out = fan_out // n
        std = gain * math.sqrt(2.0 / (fan_in + fan_out))
        a = math.sqrt(3.0) * std
        nn.init.uniform_(w, -a, a)


def initialize_model_(model):

    for name, p in model.named_parameters():
        if "lstm" in name and len(p.shape) > 1:
            xavier_uniform_n_(p)
        elif len(p.shape) > 1:
            torch.nn.init.xavier_uniform_(p)
        elif "bias" in name:
            torch.nn.init.constant_(p, 0.)


def build_model(vocab, t2i):
    vocab_size = len(vocab.w2i)
    output_size = len(t2i)

    emb_size = 300
    hidden_size = 150
    dropout = 0.5
    layer = "lstm"
    dependent_z = False

    selection = 0.3
    lasso = 0.0

    assert 0 < selection <= 1.0, "selection must be in (0, 1]"

    lambda_init = 1e-4
    lagrange_lr = 0.01
    lagrange_alpha = 0.99
    return LatentRationaleModel(
        vocab_size=vocab_size, emb_size=emb_size,
        hidden_size=hidden_size, output_size=output_size,
        vocab=vocab, dropout=dropout, layer=layer,
        dependent_z=dependent_z,
        selection=selection, lasso=lasso,
        lambda_init=lambda_init,
        lagrange_lr=lagrange_lr, lagrange_alpha=lagrange_alpha)

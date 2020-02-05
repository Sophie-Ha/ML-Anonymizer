import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import time
import datetime
import random

from evaluation.metrics import MetricHelper

MAX_LENGTH = 20


class EncoderBasic(nn.Module):

    def __init__(self, embeddings, num_layers, device):
        """

        :param Tensor embeddings: weights for the embeddings: (input_size, embedding_dim)
        :param int num_layers: number of recurrent layers
        """
        super(EncoderBasic, self).__init__()
        self.embeddings = nn.Embedding.from_pretrained(embeddings).float()
        self.hidden_size = self.embeddings.embedding_dim
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, num_layers=num_layers)
        self.num_layers = num_layers
        self.device = device

    def forward_step(self, input, hidden):
        embedded = self.embeddings(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def forward(self, input_tensor, max_length=MAX_LENGTH):
        """

        :param input_tensor: tensor of token ids
        :param max_length: maximum allowed length of input sequence
        :return:
        """
        # encoder_outputs = torch.zeros(input_tensor.size(0), self.hidden_size)
        output_list = []
        input_tensor = input_tensor.to(self.device)
        input_length = input_tensor.size(0)
        encoder_hidden = self.init_hidden()

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.forward_step(input_tensor[ei], encoder_hidden)
            # encoder_outputs[ei] = encoder_output[0, 0]
            output_list.append(encoder_output[0, 0])

        encoder_outputs = torch.stack(output_list)

        return encoder_outputs, encoder_hidden

    def init_hidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size, device=self.device)


class Attention(nn.Module):
    """ adapted from https://bastings.github.io/annotated_encoder_decoder/ """

    def __init__(self, hidden_size, input_dim):
        super(Attention, self).__init__()

        self.key_layer = nn.Linear(input_dim, hidden_size, bias=False)  # for encoder hidden states
        self.query_layer = nn.Linear(input_dim, hidden_size, bias=False)  # for decoder hidden states
        self.energy_layer = nn.Linear(hidden_size, 1, bias=False)  # combines the previous

        self.alphas = None

    def forward(self, query=None, proj_key=None):
        """
        computes attention at every decoding step
        :param query: current decoder state
        :param proj_key: pre-computed transformations of encoder states
        :return:
        """

        query = self.query_layer(query)

        scores = self.energy_layer(torch.tanh(query + proj_key))
        scores = scores.squeeze(2).unsqueeze(1)

        alphas = F.softmax(scores, dim=-1)
        self.alphas = alphas

        # alphas shape: [B, 1, M]
        return alphas


class DecoderBasic(nn.Module):

    def __init__(self, embeddings, attn_size, num_layers, device):
        super(DecoderBasic, self).__init__()
        self.embeddings = nn.Embedding.from_pretrained(embeddings).float()
        self.hidden_size = self.embeddings.embedding_dim
        self.attn_size = attn_size
        self.attention = Attention(self.attn_size, self.hidden_size)
        self.rnn = nn.GRU(self.hidden_size, self.hidden_size, num_layers)
        self.num_layers = num_layers
        self.device = device

    def forward_step(self, prev_output, decoder_hidden, proj_key):
        """

        :param prev_output: index of token previously pointed at
        :param decoder_hidden: current hidden state of the decoder
        :param proj_key: pre-computed intermediate attention values based on the encoder outputs
        :return: new hidden state and probability distribution over the inputs
        """
        prev_embed = self.embeddings(prev_output).view(1, 1, -1)  # why?

        # compute attention probabilities used as a pointer
        query = decoder_hidden[-1].unsqueeze(1)
        attn_probs = self.attention(query=query, proj_key=proj_key)

        # update rnn hidden state
        output, hidden = self.rnn(prev_embed, decoder_hidden)

        return hidden, attn_probs

    def forward(self, encoder_outputs, encoder_hidden, encoder_input, sos_token, target_tensor=None,
                max_length=MAX_LENGTH, train=True, teacher_forcing=0.5):

        hidden = self.init_hidden(encoder_hidden)

        # pre-compute keys for attention
        proj_key = self.attention.key_layer(encoder_outputs)

        decoder_states = []
        attention_outputs = []

        # unroll the decoder RNN
        hidden, attention = self.forward_step(prev_output=torch.tensor(sos_token, device=self.device),
                                              decoder_hidden=hidden, proj_key=proj_key)
        decoder_states.append(hidden)
        attention_outputs.append(attention)
        if train:
            target_tensor = target_tensor.to(self.device)
            for i in range(target_tensor.size(0)):
                if random.random() < teacher_forcing:
                    prev_output = target_tensor[i]
                else:
                    prev_output = encoder_input[torch.argmax(attention)]

                hidden, attention = self.forward_step(prev_output=prev_output, decoder_hidden=hidden,
                                                      proj_key=proj_key)
                decoder_states.append(hidden)
                attention_outputs.append(attention)
        else:
            eos_token = encoder_input[-1]
            counter = 0
            prev_output = encoder_input[torch.argmax(attention).item()]
            while counter < max_length and prev_output != eos_token:
                hidden, attention = self.forward_step(prev_output=prev_output, decoder_hidden=hidden,
                                                      proj_key=proj_key)
                prev_output = encoder_input[torch.argmax(attention)]
                decoder_states.append(hidden)
                attention_outputs.append(attention)
                counter += 1

        return decoder_states, attention_outputs

    def init_hidden(self, encoder_hidden=None):
        if encoder_hidden is not None:
            return encoder_hidden
        return torch.zeros(self.num_layers, 1, self.hidden_size, device=self.device)


class BasicModel:

    def __init__(self, embeddings, sos_token, device, attn_size=50, learning_rate=0.1, lr_factor=None, lr_patience=None,
                 n_encoder_layers=1, n_decoder_layers=1):

        self.device = device
        self.encoder = EncoderBasic(embeddings=embeddings, num_layers=n_encoder_layers, device=device)
        self.encoder.to(self.device)
        self.decoder = DecoderBasic(embeddings=embeddings, attn_size=attn_size, num_layers=n_decoder_layers,
                                    device=device)
        self.decoder.to(self.device)
        self.encoder_optimizer = optim.SGD(self.encoder.parameters(), lr=learning_rate)
        self.decoder_optimizer = optim.SGD(self.decoder.parameters(), lr=learning_rate)
        if lr_factor and lr_patience:
            self.lr_scheduling = True
            self.encoder_lr_scheduler = ReduceLROnPlateau(self.encoder_optimizer, mode="max", factor=lr_factor,
                                                          patience=lr_patience)
            self.decoder_lr_scheduler = ReduceLROnPlateau(self.decoder_optimizer, mode="max", factor=lr_factor,
                                                          patience=lr_patience)
            self.lr_factor = lr_factor
            self.lr_patience = lr_patience
        else:
            self.lr_scheduling = False
        self.sos_token = sos_token
        self.losses = []
        self.cur_epoch = 0
        self.train_time = 0
        self.lr = learning_rate

    def train(self, inputs, targets, epochs, print_interval=500, teacher_forcing=0.5, dev_inputs=None, dev_targets=None):
        """

        :param List[List[int]] inputs: tokenized input sentences
        :param List[List[int]] targets: target indices
        :param int epochs: number of epochs
        :param print_interval:
        :param teacher_forcing: teacher forcing ration (whether decoder receives previous output or previous target element
        :param dev_inputs: tokenized input sentence from the dev dataset, needed to compute lr decay
        :param dev_targets: target indices corresponding to the dev data, needed to compute lr decay
        :return:
        """
        t = time.time()

        criterion = nn.CrossEntropyLoss()

        for e in range(epochs):

            self.cur_epoch += 1

            shuffle_list = list(zip(inputs, targets))
            random.shuffle(shuffle_list)
            inputs = [a[0] for a in shuffle_list]
            targets = [a[1] for a in shuffle_list]

            for i in range(len(inputs)):
                self.losses.append(self.train_single(input_tensor=torch.tensor(inputs[i], device=self.device),
                                                     target_tensor=torch.tensor(targets[i], device=self.device),
                                                     criterion=criterion,
                                                     teacher_forcing=teacher_forcing))
                if i % print_interval == 0:
                    print(i, "iterations. avg loss:", np.mean(self.losses[-print_interval:]))

            if self.lr_scheduling:
                if dev_inputs is None or dev_targets is None:
                    raise ValueError("No Dev data for lr scheduling provided")
                f1 = self._get_dev_metric(inputs=dev_inputs, targets=dev_targets)
                self.encoder_lr_scheduler.step(f1)
                self.decoder_lr_scheduler.step(f1)
                print("Finished Epoch {}. Avg loss: {}. F1: {}. Total time: {}.".format(self.cur_epoch,
                                                                                        np.mean(self.losses[-len(inputs):]),
                                                                                        f1,
                                                                                        self.train_time + time.time()-t))
            else:
                print("Finished Epoch {}. Avg loss: {}. Total time: {}.".format(self.cur_epoch,
                                                                                np.mean(self.losses[-len(inputs):]),
                                                                                self.train_time + time.time()-t))

        self.train_time += time.time() - t
        print("Total training time:", self.train_time)

    def _get_dev_metric(self, inputs, targets):
        with torch.no_grad():
            predicted = []
            target_anno = []
            for input_sent, t in zip(inputs, targets):
                res_indices = self.predict(input_sent)
                cur_predicted = []
                cur_anno = []
                for i in range(len(input_sent) - 1):
                    if i in res_indices:
                        cur_predicted.append(1)
                    else:
                        cur_predicted.append(0)
                    if i in t:
                        cur_anno.append(1)
                    else:
                        cur_anno.append(0)
                predicted.append(cur_predicted)
                target_anno.append(cur_anno)
        mh = MetricHelper(predicted=predicted, target=target_anno)
        return mh.f1()

    def train_single(self, input_tensor, target_tensor, criterion, teacher_forcing):
        loss_sum = 0

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        target_tokens = []
        for t in target_tensor:
            target_tokens.append(input_tensor[t])

        encoder_outputs, last_encoder_hidden = self.encoder(input_tensor)
        decoder_states, attention_output = self.decoder(encoder_outputs=encoder_outputs,
                                                        encoder_hidden=last_encoder_hidden,
                                                        encoder_input=input_tensor,
                                                        target_tensor=torch.tensor(target_tokens, device=self.device),
                                                        sos_token=self.sos_token,
                                                        teacher_forcing=teacher_forcing)

        for a in range(target_tensor.size(0)):
            loss = criterion(attention_output[a].view(1, -1), target_tensor[a].unsqueeze_(dim=0).to(self.device))
            loss.backward(retain_graph=True)
            loss_sum = loss + loss_sum

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss_sum.item()

    def predict(self, input_tensor):
        with torch.no_grad():
            encoder_outputs, last_encoder_hidden = self.encoder(torch.tensor(input_tensor, device=self.device))
            decoder_states, attention_output = self.decoder(encoder_outputs=encoder_outputs,
                                                            encoder_hidden=last_encoder_hidden,
                                                            encoder_input=torch.tensor(input_tensor, device=self.device),
                                                            train=False,
                                                            sos_token=self.sos_token)
            res_indices = []
            for att in attention_output:
                res_indices.append(torch.argmax(att).item())
            res_indices = list(set(res_indices))
            res_indices.sort()
            return res_indices

    def save_model_parameters(self, path):  # TODO save less stuff
        if self.lr_scheduling:
            torch.save({"epoch": self.cur_epoch,
                        "encoder_state_dict": self.encoder.state_dict(),
                        "encoder_optim_state_dict": self.encoder_optimizer.state_dict(),
                        "encoder_lr_state_dict": self.encoder_lr_scheduler.state_dict(),
                        "decoder_state_dict": self.decoder.state_dict(),
                        "decoder_optim_state_dict": self.decoder_optimizer.state_dict(),
                        "decoder_lr_state_dict": self.decoder_lr_scheduler.state_dict(),
                        "sos_token": self.sos_token,
                        "losses": self.losses,
                        "train_time": self.train_time,
                        "datetime": datetime.datetime.now(),
                        "learning_rate": self.lr,
                        "lr_scheduling": True,
                        "lr_factor": self.lr_factor,
                        "lr_patience": self.lr_patience}, path)
        else:
            torch.save({"epoch": self.cur_epoch,
                        "encoder_state_dict": self.encoder.state_dict(),
                        "encoder_optim_state_dict": self.encoder_optimizer.state_dict(),
                        "decoder_state_dict": self.decoder.state_dict(),
                        "decoder_optim_state_dict": self.decoder_optimizer.state_dict(),
                        "sos_token": self.sos_token,
                        "losses": self.losses,
                        "train_time": self.train_time,
                        "datetime": datetime.datetime.now(),
                        "learning_rate": self.lr,
                        "lr_scheduling": False}, path)

    def load_model_parameters(self, path):
        print("Loading model from", path)
        checkpoint = torch.load(path)

        # loading model state dicts
        self.encoder.load_state_dict(checkpoint["encoder_state_dict"])
        self.encoder_optimizer.load_state_dict(checkpoint["encoder_optim_state_dict"])
        self.decoder.load_state_dict(checkpoint["decoder_state_dict"])
        self.decoder_optimizer.load_state_dict(checkpoint["decoder_optim_state_dict"])

        # load lr_scheduling if present
        if "lr_scheduling" in checkpoint:
            self.lr = checkpoint["learning_rate"]
            self.lr_scheduling = checkpoint["lr_scheduling"]
            if checkpoint["lr_scheduling"]:
                self.encoder_lr_scheduler.load_state_dict(checkpoint["encoder_lr_state_dict"])
                self.decoder_lr_scheduler.load_state_dict(checkpoint["decoder_lr_state_dict"])
                self.lr_factor = checkpoint["lr_factor"]
                self.lr_patience = checkpoint["lr_patience"]
        else:
            self.lr_scheduling = False

        # loading other parameters
        self.cur_epoch = checkpoint["epoch"]
        self.sos_token = checkpoint["sos_token"]
        self.losses = checkpoint["losses"]
        self.train_time = checkpoint["train_time"]
        print("Done loading:\nModel was saved on", checkpoint["datetime"],
              "\nNumber of epochs so far:", checkpoint["epoch"],
              "\nTime trained so far:", checkpoint["train_time"])



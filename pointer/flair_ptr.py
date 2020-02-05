import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from flair.embeddings import WordEmbeddings, FlairEmbeddings, StackedEmbeddings, BertEmbeddings, Embeddings
from flair.data import Sentence
import numpy as np
import time
import datetime
import random

from evaluation.metrics import MetricHelper

MAX_LENGTH = 20


class EncoderFlair(nn.Module):

    def __init__(self, num_layers, hidden_size, device):
        """

        :param int num_layers: number of recurrent layers
        """
        super(EncoderFlair, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, num_layers=num_layers)
        self.num_layers = num_layers
        self.device = device

    def forward_step(self, embedded, hidden):
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def forward(self, input_sentence, max_length=MAX_LENGTH):
        """

        :param Sentence input_sentence: flair sentence object containing the input
        :param max_length: maximum allowed length of input sequence
        :return:
        """
        output_list = []
        input_length = len(input_sentence.tokens) + 1
        input_embedded = [t.embedding.to(self.device) for t in input_sentence] + [torch.zeros(1, 1, self.hidden_size,
                                                                                              device=self.device)]
        encoder_hidden = self.init_hidden()

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.forward_step(input_embedded[ei].view(1, 1, -1), encoder_hidden)
            output_list.append(encoder_output[0, 0])

        encoder_outputs = torch.stack(output_list)

        return encoder_outputs, encoder_hidden

    def init_hidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size, device=self.device)


class Attention(nn.Module):

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


class DecoderFlair(nn.Module):

    def __init__(self, attn_size, num_layers, hidden_size, embedding_size, device):
        """

        :param attn_size:
        """
        super(DecoderFlair, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.attn_size = attn_size
        self.attention = Attention(self.attn_size, self.hidden_size)
        self.rnn = nn.GRU(self.hidden_size, self.hidden_size, num_layers)
        self.num_layers = num_layers
        self.device = device

    def forward_step(self, prev_output, decoder_hidden, proj_key):
        """

        :param torch.Tensor prev_output: embedding of the token previously pointed at
        :param decoder_hidden: current hidden state of the decoder
        :param proj_key: pre-computed intermediate attention values based on the encoder outputs
        :return: new hidden state and probability distribution over the inputs
        """

        # compute attention probabilities used as a pointer
        query = decoder_hidden[-1].unsqueeze(1)
        attn_probs = self.attention(query=query, proj_key=proj_key)

        # update rnn hidden state
        output, hidden = self.rnn(prev_output.view(1, 1, -1), decoder_hidden)

        return hidden, attn_probs

    def forward(self, encoder_outputs, encoder_hidden, input_sentence, target_tensor=None,
                max_length=MAX_LENGTH, train=True, teacher_forcing=0.5):
        """

        :param encoder_outputs:
        :param torch.Tensor encoder_hidden: Last hidden state of the encoder
        :param Sentence input_sentence:
        :param torch.Tensor target_tensor: tensor of target word indices
        :param max_length:
        :param train:
        :param teacher_forcing:
        :return:
        """

        hidden = self.init_hidden(encoder_hidden)
        input_embedded = [t.embedding.to(self.device) for t in input_sentence] + [torch.zeros(1, 1, self.embedding_size,
                                                                                              device=self.device)]

        # pre-compute keys for attention
        proj_key = self.attention.key_layer(encoder_outputs)

        decoder_states = []
        attention_outputs = []

        sos = torch.zeros(1, 1, self.embedding_size, device=self.device)

        # unroll the decoder RNN
        hidden, attention = self.forward_step(prev_output=sos, decoder_hidden=hidden,
                                              proj_key=proj_key)
        decoder_states.append(hidden)
        attention_outputs.append(attention)
        if train:
            for i in range(target_tensor.size(0)):
                if random.random() < teacher_forcing:
                    prev_index = target_tensor[i]
                    prev_output = input_embedded[prev_index]
                else:
                    prev_index = torch.argmax(attention)
                    prev_output = input_embedded[prev_index]

                hidden, attention = self.forward_step(prev_output=prev_output, decoder_hidden=hidden,
                                                      proj_key=proj_key)
                decoder_states.append(hidden)
                attention_outputs.append(attention)
        else:
            counter = 0
            prev_output = input_embedded[torch.argmax(attention)]
            while counter < max_length and not torch.all(torch.eq(prev_output, sos)):
                hidden, attention = self.forward_step(prev_output=prev_output, decoder_hidden=hidden,
                                                      proj_key=proj_key)
                prev_output = input_embedded[torch.argmax(attention)]
                decoder_states.append(hidden)
                attention_outputs.append(attention)
                counter += 1

        return decoder_states, attention_outputs

    def init_hidden(self, encoder_hidden=None):
        if encoder_hidden is not None:
            return encoder_hidden
        return torch.zeros(self.num_layers, 1, self.hidden_size, device=self.device)


class FlairPointer:

    def __init__(self, hidden_size, embedding_size, attn_size=50, learning_rate=0.1, lr_factor=None, lr_patience=None,
                 n_encoder_layers=1, n_decoder_layers=1, cuda_device=0):

        self.device = torch.device("cuda:" + str(cuda_device) if torch.cuda.is_available() else "cpu")
        self.encoder = EncoderFlair(num_layers=n_encoder_layers, hidden_size=hidden_size, device=self.device)
        self.decoder = DecoderFlair(attn_size=attn_size, num_layers=n_decoder_layers, hidden_size=hidden_size,
                                    embedding_size=embedding_size, device=self.device)
        self.encoder.to(self.device)
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
        self.losses = []
        self.cur_epoch = 0
        self.train_time = 0
        self.lr = learning_rate

    def train(self, inputs, targets, epochs, print_interval=500, teacher_forcing=0.5, dev_inputs=None, dev_targets=None,
              print_info=False):
        """

        :param List[Sentence] inputs: tokenized input sentences that have already been embedded
        :param List[List[int]] targets: target indices
        :param int epochs: number of epochs
        :param print_interval:
        :param teacher_forcing: teacher forcing ration (whether decoder receives previous output or previous target element
        :param List[Sentence] dev_inputs: tokenized input sentence from the dev dataset, needed to compute lr decay
        :param List[List[int]] dev_targets: target indices corresponding to the dev data, needed to compute lr decay
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
                self.losses.append(self.train_single(input_sentence=inputs[i],
                                                     target_tensor=torch.tensor(targets[i], device=self.device),
                                                     criterion=criterion,
                                                     teacher_forcing=teacher_forcing))
                if print_info and i % print_interval == 0:
                    print(i, "iterations. avg loss:", np.mean(self.losses[-print_interval:]))

            if self.lr_scheduling:
                if dev_inputs is None or dev_targets is None:
                    raise ValueError("No Dev data for lr scheduling provided")
                f1 = self._get_dev_metric(inputs=dev_inputs, targets=dev_targets)
                self.encoder_lr_scheduler.step(f1)
                self.decoder_lr_scheduler.step(f1)
                print("Finished Epoch {}. Avg loss: {}. F1: {}. Epoch Time: {}. Total time: {}.".format(self.cur_epoch,
                                                                                                        np.mean(self.losses[-len(inputs):]),
                                                                                                        f1,
                                                                                                        time.time() - t,
                                                                                                        self.train_time + time.time()-t))
            else:
                print("Finished Epoch {}. Avg loss: {}. Epoch Time: {}. Total time: {}.".format(self.cur_epoch,
                                                                                                np.mean(self.losses[-len(inputs):]),
                                                                                                time.time() - t,
                                                                                                self.train_time + time.time()-t))

        self.train_time += time.time() - t
        print("Total training time:", self.train_time)

    def _get_dev_metric(self, inputs, targets):
        """

        :param list[Sentence] inputs:
        :param list[list[int]] targets:
        :return:
        """
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

    def train_single(self, input_sentence, target_tensor, criterion, teacher_forcing):
        loss_sum = 0

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        encoder_outputs, last_encoder_hidden = self.encoder(input_sentence)
        decoder_states, attention_output = self.decoder(encoder_outputs=encoder_outputs,
                                                        encoder_hidden=last_encoder_hidden,
                                                        input_sentence=input_sentence,
                                                        target_tensor=target_tensor,
                                                        teacher_forcing=teacher_forcing)

        # target_tensor = target_tensor.to(self.device)
        for a in range(target_tensor.size(0)):
            loss = criterion(attention_output[a].view(1, -1), target_tensor[a].unsqueeze_(dim=0))
            loss.backward(retain_graph=True)
            loss_sum = loss + loss_sum

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss_sum.item()

    def predict(self, input_sentence):
        """

        :param Sentence input_sentence:
        :return:
        """
        with torch.no_grad():
            encoder_outputs, last_encoder_hidden = self.encoder(input_sentence)
            decoder_states, attention_output = self.decoder(encoder_outputs=encoder_outputs,
                                                            encoder_hidden=last_encoder_hidden,
                                                            input_sentence=input_sentence,
                                                            train=False)
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
        self.losses = checkpoint["losses"]
        self.train_time = checkpoint["train_time"]
        print("Done loading:\nModel was saved on", checkpoint["datetime"],
              "\nNumber of epochs so far:", checkpoint["epoch"],
              "\nTime trained so far:", checkpoint["train_time"])

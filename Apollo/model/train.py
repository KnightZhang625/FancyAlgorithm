import torch
import torch.nn as nn
import torch.optim as optim
from Utils.utils import prepare_sequence, padding_function

class Train(object):
    def __init__(self, encoder_model, decoder_model, train_x, train_y, src_idx, tar_idx, epoch, batch_size, device):
        self.encoder_model = encoder_model
        self.decoder_model = decoder_model
        self.train_x = train_x
        self.train_y = train_y
        self.src_idx = src_idx
        self.tar_idx = tar_idx
        self.epoch = epoch
        self.batch_size = batch_size
        self.device = device
    
    def launch(self, max_length):
        encoder_optimizer = optim.Adam(self.encoder_model.parameters(), lr=0.01)
        decoder_optimizer = optim.Adam(self.decoder_model.parameters(), lr=0.01)
        loss_function = nn.NLLLoss()
        for ep in range(self.epoch):
            for b_start in range(0, len(self.train_x), self.batch_size):
                batch_x = self.train_x[b_start: b_start+self.batch_size]
                batch_y = self.train_y[b_start: b_start+self.batch_size]
                x_padding, X_lengths = padding_function(batch_x, self.src_idx, self.device)
                target = prepare_sequence(batch_y, self.tar_idx, self.device)
                target = sorted(target, key=lambda t : t.size()[-1], reverse=True)
                
                encoder_outputs = self.encoder_model(x_padding, X_lengths)
                
                decoder_input = torch.zeros(self.batch_size, 1, dtype=torch.long, device=self.device)
                prev_hidden = encoder_outputs[-1, ].unsqueeze(0)
        
                for i in range(decoder_input.size()[0]):
                    decoder_input[i, :] = torch.tensor(self.tar_idx['SOS'], dtype=torch.long, device=self.device)
                
                self.max_length = max_length
                decoder_outputs = []

                for i in range(self.max_length):
                    decoder_output, prev_hidden = self.decoder_model(decoder_input, prev_hidden, encoder_outputs)
                    decoder_outputs.append(decoder_output)
                    _, topI = decoder_output.topk(1)
                    topI = topI.squeeze().view(self.batch_size, -1)
                    decoder_input = topI.clone().detach()

            loss = self.calcuate_loss(target, decoder_outputs, loss_function, self.batch_size)
            self.encoder_model.zero_grad()
            self.decoder_model.zero_grad()
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()
            print(loss.item())
    
    def calcuate_loss(self, y_true, y_pred, loss_func, batch_size):
        mask = [data.size()[-1] for data in y_true]

        y_true_for_loss = y_true[0]
        for i in range(1, len(y_true)):
            y_true_for_loss = torch.cat((y_true_for_loss, y_true[i]))
        
        y_pred_for_loss = y_pred[0][0,]
        for i in range(batch_size):
            if i == 0:
                for j in range(1, mask[i]):
                    y_pred_for_loss = torch.cat((y_pred_for_loss, y_pred[j][i]))
            else:
                for j in range(0, mask[i]):
                    y_pred_for_loss = torch.cat((y_pred_for_loss, y_pred[j][i]))
        loss = loss_func(y_pred_for_loss, y_true_for_loss)
        return loss
    
    def predict(self, data, idx_src, idx_tar):
        x_padding, X_lengths = padding_function(data, self.src_idx, self.device)
        encoder_outputs = self.encoder_model(x_padding, X_lengths)
        decoder_input = torch.zeros(self.batch_size, 1, dtype=torch.long, device=self.device)
        prev_hidden = encoder_outputs[-1, ].unsqueeze(0)
        for i in range(decoder_input.size()[0]):
            decoder_input[i, :] = torch.tensor(self.tar_idx['SOS'], dtype=torch.long, device=self.device)

        decoder_outputs = []
        results = []
        for i in range(self.max_length):
            '''
                decoder_output : (2, 1, 11)
                prev_hidden : (1, 2, 4)
            '''
            decoder_output, prev_hidden = self.decoder_model(decoder_input, prev_hidden, encoder_outputs)
            decoder_outputs.append(decoder_output)
            _, topI = decoder_output.topk(1)
            results.append(topI)
            topI = topI.squeeze().view(self.batch_size, -1)
            decoder_input = topI.clone().detach()
            
        re_1 = [idx_tar[i[0].item()] for i in results]
        re_2 = [idx_tar[i[1].item()] for i in results]
        print('src :', [idx_src[i.item()] for i in x_padding[0,]])
        print('tar :', re_1)
        print('src :', [idx_src[i.item()] for i in x_padding[1,]])
        print('tar :', re_2)
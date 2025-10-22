from attention_model import TransformerBlock
import torch
import torch.nn as nn
import torch.nn.functional as F

class Multi_scale_CNN_attention(nn.Module):
    def __init__(self, args):
        super(Multi_scale_CNN_attention, self).__init__()
        self.CNN_branch_1 = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(.2),
            nn.Dropout(args.drop_out_prob),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(.2),
            nn.Dropout(args.drop_out_prob),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(.2),
            nn.Dropout(args.drop_out_prob),
            nn.MaxPool1d(kernel_size=3, stride=2)
            # nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm1d(256),
            # nn.LeakyReLU(.2),
            # nn.Dropout(args.drop_out_prob),
            # nn.MaxPool1d(kernel_size=3, stride=2),
            # nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm1d(512),
            # nn.LeakyReLU(.2),
            # nn.Dropout(args.drop_out_prob),
            # nn.MaxPool1d(kernel_size=3, stride=2)
        )

        self.CNN_branch_2 = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(.2),
            nn.Dropout(args.drop_out_prob),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(.2),
            nn.Dropout(args.drop_out_prob),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(.2),
            nn.Dropout(args.drop_out_prob),
            nn.MaxPool1d(kernel_size=3, stride=2)
            # nn.Conv1d(128, 256, kernel_size=5, stride=1, padding=1),
            # nn.BatchNorm1d(256),
            # nn.LeakyReLU(.2),
            # nn.Dropout(args.drop_out_prob),
            # nn.MaxPool1d(kernel_size=3, stride=1),
            # nn.Conv1d(256, 512, kernel_size=5, stride=1, padding=1),
            # nn.BatchNorm1d(512),
            # nn.LeakyReLU(.2),
            # nn.Dropout(args.drop_out_prob),
            # nn.MaxPool1d(kernel_size=3, stride=1)
        )
        # self.featrue_dropout = nn.Dropout(0.5)

        self.CNN_branch_3 = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=7, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(.2),
            nn.Dropout(args.drop_out_prob),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(32, 64, kernel_size=7, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(.2),
            nn.Dropout(args.drop_out_prob),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(64, 128, kernel_size=7, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(.2),
            nn.Dropout(args.drop_out_prob),
            nn.MaxPool1d(kernel_size=3, stride=1)
            # nn.Conv1d(128, 256, kernel_size=5, stride=1, padding=1),
            # nn.BatchNorm1d(256),
            # nn.LeakyReLU(.2),
            # nn.Dropout(args.drop_out_prob),
            # nn.MaxPool1d(kernel_size=3, stride=1),
            # nn.Conv1d(256, 512, kernel_size=5, stride=1, padding=1),
            # nn.BatchNorm1d(512),
            # nn.LeakyReLU(.2),
            # nn.Dropout(args.drop_out_prob),
            # nn.MaxPool1d(kernel_size=3, stride=1)
        )
        self.MS_attn = nn.Sequential(MSCNN_atten)
        self.MS_attn2 = nn.Sequential(MSCNN_atten2)
        self.MS_attn3 = nn.Sequential(MSCNN_atten3)
        self.flatten = nn.Flatten(start_dim=-2, end_dim=-1)
        self.fc = nn.Sequential(
            nn.Linear(4480, 256),
            nn.LeakyReLU(.2),
            nn.Linear(256, args.class_num)
        )
    def forward(self, x):
        branch_1 = self.CNN_branch_1(x)
        branch_2 = self.CNN_branch_2(x)
        branch_3 = self.CNN_branch_3(x)
        branch_1 = torch.flatten(branch_1, start_dim=-2, end_dim=-1)
        branch_2 = torch.flatten(branch_2, start_dim=-2, end_dim=-1)
        branch_3 = torch.flatten(branch_3, start_dim=-2, end_dim=-1)
        fusion_featrue = torch.cat((branch_1, branch_2,branch_3), dim=1)
        fusion_featrue = self.MS_attn(fusion_featrue)
        fusion_featrue = self.MS_attn2(fusion_featrue)
        fusion_featrue = self.MS_attn3(fusion_featrue)
        fusion_featrue = self.fc(fusion_featrue)
        # fusion_featrue = self.featrue_dropout(fusion_featrue)
            # if(self.opts["featrue_bottleneck"]):
            #     return fusion_featrue
            # else:
            #     return self.bottle_1(fusion_featrue)
        del branch_1, branch_2
        torch.cuda.empty_cache()
        return fusion_featrue

b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, padding=(1, 0), stride=2))

MSCNN_atten = TransformerBlock(input_dim=4480, num_heads=5, ff_dim=1500, dropout=0.2)
MSCNN_atten2 = TransformerBlock(input_dim=4480, num_heads=5, ff_dim=1500, dropout=0.2)
MSCNN_atten3 = TransformerBlock(input_dim=4480, num_heads=5, ff_dim=1500, dropout=0.2)


attn = TransformerBlock(input_dim=100, num_heads=5, ff_dim=1500, dropout=0.2)
attn_block2 = TransformerBlock(input_dim=100, num_heads=5, ff_dim=1500, dropout=0.2)
attn_block3 = TransformerBlock(input_dim=100, num_heads=5, ff_dim=1500, dropout=0.2)
class My_Transfromer(nn.Module):
    def __init__(self, args):
        super(My_Transfromer, self).__init__()
        # self.pre_part = nn.Sequential(b1)
        self.flatten = nn.Flatten(start_dim=-2, end_dim=-1)
        self.attn = nn.Sequential(attn)
        self.attn2 = nn.Sequential(attn_block2)
        # self.attn3 = nn.Sequential(attn_block3)


        self.linear = nn.Linear(200, args.class_num)
    def forward(self, x):
        # output = self.pre_part(x)
        #
        # output = self.flatten(output)
        output = self.attn(x)
        output = self.attn2(output)
        # output = self.attn3(output)
        output = self.flatten(output)
        output = self.linear(output)
        return output

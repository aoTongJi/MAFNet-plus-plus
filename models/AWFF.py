class HaarWaveletTransform(nn.Module):
    def __init__(self):
        super().__init__()
        h = torch.tensor([1., 1.]) / torch.sqrt(torch.tensor(2.))
        g = torch.tensor([1., -1.]) / torch.sqrt(torch.tensor(2.))
        
        self.register_buffer('LL', torch.outer(h, h)[None, None, :, :])
        self.register_buffer('LH', torch.outer(h, g)[None, None, :, :])
        self.register_buffer('HL', torch.outer(g, h)[None, None, :, :])
        self.register_buffer('HH', torch.outer(g, g)[None, None, :, :])

    def forward(self, x):
        """ 
        x: [B, C, H, W]
        return: LL, LH, HL, HH  各为 [B, C, H/2, W/2]
        """
        LL = F.conv2d(x, self.LL.expand(x.size(1), -1, -1, -1), stride=2, groups=x.size(1))
        LH = F.conv2d(x, self.LH.expand(x.size(1), -1, -1, -1), stride=2, groups=x.size(1))
        HL = F.conv2d(x, self.HL.expand(x.size(1), -1, -1, -1), stride=2, groups=x.size(1))
        HH = F.conv2d(x, self.HH.expand(x.size(1), -1, -1, -1), stride=2, groups=x.size(1))
        return LL, LH, HL, HH


class WaveletFrequencyAttention(nn.Module):
    def __init__(self, cv_chan, feat_chan):
        super().__init__()
        self.wavelet = HaarWaveletTransform()

        self.high_att_conv = nn.Sequential(
            nn.Conv2d(feat_chan, feat_chan // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_chan // 2, cv_chan, 1),
            nn.Sigmoid()
        )
        self.low_att_conv = nn.Sequential(
            nn.Conv2d(feat_chan, feat_chan // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_chan // 2, cv_chan, 1),
            nn.Sigmoid()
        )

        self.fusion_att = nn.Sequential(
            nn.Conv2d(cv_chan * 2, cv_chan, 1),
            nn.Sigmoid()
        )

        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(feat_chan, feat_chan // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_chan // 8, feat_chan, 1),
            nn.Sigmoid()
        )
        self.channel_proj = nn.Conv2d(feat_chan, cv_chan, kernel_size=1)


    def forward(self, cv, feat):
        """
        cv: [B, C, D, H, W]
        feat: [B, C_feat, H, W]
        """
        B, C, D, H, W = cv.shape
  
        LL, LH, HL, HH = self.wavelet(feat)
        high_freq = F.interpolate((LH + HL + HH) / 3.0, size=(H, W), mode='bilinear', align_corners=False)
        low_freq = F.interpolate(LL, size=(H, W), mode='bilinear', align_corners=False)
        
        high_att = self.high_att_conv(high_freq).unsqueeze(2)  
        low_att = self.low_att_conv(low_freq).unsqueeze(2)     
        
        cv_high = high_att * cv
        cv_low = low_att * cv
 
        cv_cat = torch.cat([cv_high, cv_low], dim=1)            # [B, 2C, D, H, W]
        cv_cat_2d = cv_cat.view(B, 2*C, D*H, W)
        fuse_weight = self.fusion_att(cv_cat_2d).view(B, C, D, H, W)
        cv_fused = fuse_weight * cv_high + (1 - fuse_weight) * cv_low
        
        ch_att = self.channel_att(feat)                         # [B, C_feat, 1, 1]
        ch_att = torch.sigmoid(self.channel_proj(ch_att)).unsqueeze(2)       # [B, C, 1, 1, 1]
        cv_final = cv_fused * ch_att
        
        return cv_final
class ShiftedWindowCrossAttention(nn.Module):

    
    def __init__(self, stereo_dim: int, mono_dim: int, hidden_dim: int = 256, 
                 num_heads: int = 8, window_size: int = 8, shift_ratio: float = 0.5,
                 dropout: float = 0.1):
        super().__init__()
        
        self.stereo_dim = stereo_dim
        self.mono_dim = mono_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_ratio = shift_ratio
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.stereo_q = nn.Linear(stereo_dim, hidden_dim, bias=False)
        self.stereo_k = nn.Linear(stereo_dim, hidden_dim, bias=False)
        self.stereo_v = nn.Linear(stereo_dim, hidden_dim, bias=False)
        self.stereo_proj = nn.Linear(hidden_dim, stereo_dim)
        
        self.mono_q = nn.Linear(mono_dim, hidden_dim, bias=False)
        self.mono_k = nn.Linear(mono_dim, hidden_dim, bias=False)
        self.mono_v = nn.Linear(mono_dim, hidden_dim, bias=False)
        self.mono_proj = nn.Linear(hidden_dim, mono_dim)
        
        self.norm_stereo = nn.LayerNorm(stereo_dim)
        self.norm_mono = nn.LayerNorm(mono_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = self.head_dim ** -0.5
        
    def create_windows(self, x: torch.Tensor, shift: bool = False) -> Tuple[torch.Tensor, Tuple[int, int]]:

        B, C, H, W = x.shape
        
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
            
        H_pad, W_pad = H + pad_h, W + pad_w
        
        if shift:
            shift_h = int(self.window_size * self.shift_ratio)
            shift_w = int(self.window_size * self.shift_ratio)
            x = torch.roll(x, shifts=(-shift_h, -shift_w), dims=(2, 3))
        
        x = x.view(B, C, H_pad // self.window_size, self.window_size, 
                   W_pad // self.window_size, self.window_size)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        x = x.view(-1, self.window_size * self.window_size, C)
        
        return x, (H_pad, W_pad)
    
    def merge_windows(self, windows: torch.Tensor, H_pad: int, W_pad: int, 
                     shift: bool = False) -> torch.Tensor:
        
        B = windows.shape[0] // (H_pad // self.window_size * W_pad // self.window_size)
        C = windows.shape[-1]
        
        windows = windows.view(B, H_pad // self.window_size, W_pad // self.window_size,
                              self.window_size, self.window_size, C)
        windows = windows.permute(0, 5, 1, 3, 2, 4).contiguous()
        windows = windows.view(B, C, H_pad, W_pad)
        
        if shift:
            shift_h = int(self.window_size * self.shift_ratio)
            shift_w = int(self.window_size * self.shift_ratio)
            windows = torch.roll(windows, shifts=(shift_h, shift_w), dims=(2, 3))
        
        return windows
    
    def forward(self, stereo_feat: torch.Tensor, mono_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        B, C_stereo, H, W = stereo_feat.shape
        _, C_mono, _, _ = mono_feat.shape

        orig_stereo = stereo_feat
        orig_mono = mono_feat
        
        stereo_feat = self.norm_stereo(stereo_feat.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        mono_feat = self.norm_mono(mono_feat.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        stereo_enhanced = None
        mono_enhanced = None
        
        for shift in [False, True]:
            stereo_windows, (H_pad, W_pad) = self.create_windows(stereo_feat, shift)
            mono_windows, _ = self.create_windows(mono_feat, shift)
            
            stereo_q = self.stereo_q(stereo_windows).view(-1, self.window_size * self.window_size, 
                                                         self.num_heads, self.head_dim).transpose(1, 2)
            stereo_k = self.stereo_k(stereo_windows).view(-1, self.window_size * self.window_size,
                                                         self.num_heads, self.head_dim).transpose(1, 2)
            stereo_v = self.stereo_v(stereo_windows).view(-1, self.window_size * self.window_size,
                                                          self.num_heads, self.head_dim).transpose(1, 2)
            
            mono_q = self.mono_q(mono_windows).view(-1, self.window_size * self.window_size,
                                                    self.num_heads, self.head_dim).transpose(1, 2)
            mono_k = self.mono_k(mono_windows).view(-1, self.window_size * self.window_size,
                                                   self.num_heads, self.head_dim).transpose(1, 2)
            mono_v = self.mono_v(mono_windows).view(-1, self.window_size * self.window_size,
                                                   self.num_heads, self.head_dim).transpose(1, 2)
            
            stereo_attn = torch.matmul(stereo_q, mono_k.transpose(-2, -1)) * self.scale
            stereo_attn = F.softmax(stereo_attn, dim=-1)
            stereo_attn = self.dropout(stereo_attn)
            stereo_out = torch.matmul(stereo_attn, mono_v)
            
            mono_attn = torch.matmul(mono_q, stereo_k.transpose(-2, -1)) * self.scale
            mono_attn = F.softmax(mono_attn, dim=-1)
            mono_attn = self.dropout(mono_attn)
            mono_out = torch.matmul(mono_attn, stereo_v)
            

            stereo_out = stereo_out.transpose(1, 2).contiguous().view(-1, self.window_size * self.window_size, self.hidden_dim)
            mono_out = mono_out.transpose(1, 2).contiguous().view(-1, self.window_size * self.window_size, self.hidden_dim)
            
            stereo_out = self.stereo_proj(stereo_out)
            mono_out = self.mono_proj(mono_out)
            
 
            stereo_out = self.merge_windows(stereo_out, H_pad, W_pad, shift)
            mono_out = self.merge_windows(mono_out, H_pad, W_pad, shift)
            

            stereo_out = stereo_out[:, :, :H, :W]
            mono_out = mono_out[:, :, :H, :W]
            
            if stereo_enhanced is None:
                stereo_enhanced = stereo_out
                mono_enhanced = mono_out
            else:
                stereo_enhanced += stereo_out
                mono_enhanced += mono_out
        

        stereo_enhanced = stereo_enhanced / 2
        mono_enhanced = mono_enhanced / 2
        

        stereo_enhanced = stereo_enhanced + orig_stereo
        mono_enhanced = mono_enhanced + orig_mono
        

        return stereo_enhanced, mono_enhanced
